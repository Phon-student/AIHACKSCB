"""
Takes a pretrained model with classification head and uses the peft package to do Adapter + LoRA
fine tuning.
"""
from typing import Any

import torch
from lightning import LightningModule
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW, Optimizer
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_f1_score,
    binary_precision,
    binary_recall,
)
from transformers import AutoModelForSequenceClassification

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-14B-Chat-Int4")

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    task_type="CAUSAL_LM",




class Generator(LightningModule):
    def __init__(
        self,
        pretrained_model: str,
        num_classes: int,
        lr: float,
    ):
        super().__init__()

        argmodel = ModelArguments()

        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
        )
        '''
        config = transformers.AutoConfig.from_pretrained(
            argmodel.model_name_or_path,
            trust_remote_code=True,
            )
        '''

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
                argmodel.model_name_or_path,
                config=config,
                device_map ="cuda",
                trust_remote_code=True,
                quantization_config=GPTQConfig(
                    bits=4, disable_exllama=True
                )
            )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            argmodel.model_name_or_path,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token_id = 151646 # '<|extra_0|>'

        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        arg = LoraArguments()

        config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=arg.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save = ["wte", "lm_head"]
        )

        self.temperature = 0.9
        self.top_p = 0.5
        self.top_k = 3

        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, config)
        self.print_trainable_parameters(self.model)
        self.device = self.model.device

    def print_trainable_parameters(self,model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
            trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
        )

    def forward(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        labels=None
    ):
        """Calc the loss by passing inputs to the model and comparing against ground
        truth labels. Here, all of the arguments of self.model comes from the
        SequenceClassification head from HuggingFace.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    
    def single_out(self, logits):
        logits = logits['logits'][:, -1, :]
        logits = logits / self.temperature
        greedy = False
        if greedy:
            out = torch.argmax(logits, dim=1).reshape(-1, 1)
            return out

        # Initialize mask with ones
        mask = torch.ones_like(logits).bool()

        if self.top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=1), dim=1)
            sorted_mask = cumulative_probs > self.top_p
            # Ensure at least the most probable is included if sorted_mask contains all True 
            if sorted_mask.all():
                sorted_mask[..., :1] = 0
            to_scatter = sorted_mask.type_as(logits) * float('-inf')
            to_scatter[sorted_mask == 0] = logits.gather(1, sorted_indices)[sorted_mask == 0]
            logits.scatter_(1, sorted_indices, to_scatter)
        elif self.top_k > 0:
            self.top_k = min(self.top_k, logits.shape[1])            
            values, _ = torch.topk(logits, self.top_k)
            # smallest allowed value
            kth_values = values[..., -1]
            logits = torch.where(logits < kth_values.unsqueeze(-1), torch.tensor(float('-inf')).type_as(logits), logits)

        probs = torch.softmax(logits, dim=1)
        m = torch.argmax(probs,dim=1)
        return m.reshape(-1, 1)

    def gen_out(self,prompt):
        idx = tokenizer(prompt, return_tensors="pt").to(self.device)
        for _ in range(100):
            logits = model(**idx)
            next_token = single_out(logits)
            next_tokened = tokenizer.decode(next_token[0], skip_special_tokens=False)
            # print(idx.input_ids[0].shape)
            idx = tokenizer.decode(idx.input_ids[0], skip_special_tokens=False)
            
            idx = tokenizer(idx + next_tokened, return_tensors="pt",).to(self.device)

            if next_token.item() == tokenizer.eod_id:
                break
        return idx

    



#self-explained bert


class BertCls(pl.LightningModule):
    def __init__(self, num_target_classes):
        super().__init__()
        encoder = AutoModelForMaskedLM.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
        self.encoder = encoder.roberta
        #self.classification_head = Clsfication_head(num_target_classes)
        self.tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased",  model_max_length=416, padding_side="right", truncation_side='left')
        self.span_info_collect = SICModel(768)
        self.interpretation = InterpretationModel(768)
        self.output = nn.Linear(768, num_target_classes)

    def forward(self, sent_id, mask, start_indexs, end_indexs):
        x = self.encoder(input_ids=sent_id, attention_mask=mask)[0][:, 0]
        h_ij = self.span_info_collect(x, start_indexs, end_indexs)
        # interpretation layer
        H, a_ij = self.interpretation(h_ij, span_masks)
        # output layer
        out = self.output(H)
        return out, a_ij



class SICModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_1 = nn.Linear(hidden_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, hidden_size)
        self.W_3 = nn.Linear(hidden_size, hidden_size)
        self.W_4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, start_indexs, end_indexs):
        W1_h = self.W_1(hidden_states)  # (bs, length, hidden_size)
        W2_h = self.W_2(hidden_states)
        W3_h = self.W_3(hidden_states)
        W4_h = self.W_4(hidden_states)

        W1_hi_emb = torch.index_select(W1_h, 1, start_indexs)  # (bs, span_num, hidden_size)
        W2_hj_emb = torch.index_select(W2_h, 1, end_indexs)
        W3_hi_start_emb = torch.index_select(W3_h, 1, start_indexs)
        W3_hi_end_emb = torch.index_select(W3_h, 1, end_indexs)
        W4_hj_start_emb = torch.index_select(W4_h, 1, start_indexs)
        W4_hj_end_emb = torch.index_select(W4_h, 1, end_indexs)

        # [w1*hi, w2*hj, w3(hi-hj), w4(hi⊗hj)]
        span = W1_hi_emb + W2_hj_emb + (W3_hi_start_emb - W3_hi_end_emb) + torch.mul(W4_hj_start_emb, W4_hj_end_emb)
        h_ij = torch.tanh(span)
        return h_ij

class InterpretationModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.h_t = nn.Linear(hidden_size, 1)

    def forward(self, h_ij, span_masks):
        o_ij = self.h_t(h_ij).squeeze(-1)  # (ba, span_num)
        # mask illegal span
        o_ij = o_ij - span_masks
        # normalize all a_ij, a_ij sum = 1
        a_ij = nn.functional.softmax(o_ij, dim=1)
        # weight average span representation to get H
        H = (a_ij.unsqueeze(-1) * h_ij).sum(dim=1)  # (bs, hidden_size)
        return H, a_ij
