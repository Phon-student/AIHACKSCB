from typing import Callable, Iterator
from omegaconf import DictConfig
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import deque, namedtuple
from operator import itemgetter
import logging

from lightning.fabric import Fabric
import torch.nn.functional as F
import torch
import lightning as L
import numpy as np
from sklearn.metrics import f1_score
from utils import * ##


class Train_GAIL():

     def __init__(self):

        self.epochs = 10
        self.buffer_size = 100000
        self.G_batch_size = 2
        self.D_batch_size = 2
        self.grad_accum = 5

        self.ppo_epsilon = 0.2
        self.lamb = 0.8
        
        self.use_wab = False

        fabric = Fabric(accelerator="cuda", devices="auto", strategy="auto", precision="16-mixed") #need callback
        fabric.launch()

        # Two models
        self.generator = Generator()
        self.discriminator = Discriminator()

        # Two optimizers
        self.optimizer_gen = torch.optim.SGD(self.generator.parameters(), lr=0.01)
        self.optimizer_dis = torch.optim.SGD(self.discriminator.parameters(), lr=0.001)

        # Set up generator
        self.generator, self.optimizer_gen = fabric.setup(self.generator, self.optimizer_gen)
        # Set up discriminator
        self.discriminator, self.optimizer_dis = fabric.setup(self.discriminator, self.optimizer_dis)

        self.dis_train_dataloader = fabric.setup_dataloaders(dis_trains_dataloader)

    #update discriminator
    def train_discriminator_step(self, data_loader):
        self.discriminator.train
        for i in range(self.epochs):
            for batch_idx, data_batch in enumerate(data_loader.get_sample()):
                real_text, fake_text = data_batch
                real_score, reg2 = self.discriminator(**self.discriminator.tokenizer(real_text, return_tensors="pt",  max_length = 416, truncation=True, pad_to_max_length=True,))
                fake_score, reg1 = self.discriminator(**self.discriminator.tokenizer(fake_text, return_tensors="pt",  max_length = 416, truncation=True, pad_to_max_length=True,))

                real_loss = cross_entropy(tf.ones_like(real_score), real_score)
                fake_loss = cross_entropy(tf.zeros_like(fake_score), fake_score)
                reg_loss1 = self.args.lamb * reg1.pow(2).sum(dim=1).mean()
                reg_loss2 = self.args.lamb * reg2.pow(2).sum(dim=1).mean()
                reg_loss =  reg_loss2+reg_loss1
                total_loss = real_loss + fake_loss - self.lamb*reg_loss

                if (batch_idx + 1) % self.grad_accum == 0:
                    self.optimizer_dis.step()
                    self.optimizer_dis.zero_grad()
                if self.use_wab:
                    wandb.log({"D_train_loss": total_loss.item()})

    #update generator
    def train_generator_step(self, data_loader):
        self.generator.train()
        for i in range(self.epochs):
            for batch_idx, data_batch in enumerate(data_loader.get_sample()):
                prompt, old_next_tok, old_prob, normed_reward = data_batch

                tok_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, pad_to_max_length=True)
                output = self.generator(input_ids=tok_prompt['input_ids'], attention_mask=tok_prompt['attention_mask'], labels=tok_prompt['input_ids'])

                ratio = (torch.log(output['logits'][:,-1,old_next_tok]) - torch.log(old_prob)).exp()
                policy_loss1 = -normed_reward * ratio
                policy_loss2 = -normed_reward * ratio.clamp(1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon)

                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                loss = policy_loss '''+ output["loss"]'''

                if self.use_wab:
                    wandb.log({"G_train_loss": loss.item()})
                    wandb.log({"G_policy_ratio": ratio.item()})  

    def train_loop(self):
        #custom
        pass
    
