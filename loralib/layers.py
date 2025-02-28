#  ------------------------------------------------------------------------------------------
#  This code is reconstructed based on loralib (https://github.com/microsoft/LoRA) by Baijiong Lin.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
# from loralib.adalora import SVDLinear
import math
from typing import Optional, List
from collections import OrderedDict

class MultiLoRALinearLayer(nn.Linear):
    def __init__(
            self,
            existing_linear: nn.Linear,
            lora_r: int = 0,
            lora_alpha: int = 1,
            dropout_rate: float = 0.0,
            num_loras: int = 1,
            **kwargs,
    ):
        nn.Linear.__init__(self,
            in_features=existing_linear.in_features,
            out_features=existing_linear.out_features,
            **kwargs)

        self.load_state_dict(existing_linear.state_dict())

        if lora_r > 0:
            self.lora_A_list = nn.ParameterList([nn.Parameter(torch.zeros(lora_r, existing_linear.in_features))
                                                 for _ in range(num_loras)]) 

            self.lora_B_list = nn.ParameterList([nn.Parameter(torch.zeros(existing_linear.out_features, lora_r))
                              for _ in range(num_loras)])#

            self.weight.requires_grad = False
            
            self.bias.requires_grad = False

        self.init_lora_param()

        self.dropout_rate = dropout_rate
        self.r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_gate = None
        self.num_loras = num_loras

        self.before_lora_A_feature = 0
        self.after_lora_A_feature = 0

        if self.r > 0:
            self.scaling = self.lora_alpha / math.sqrt(self.r)  #

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = None

    def set_lora_gate(self, lora_gate: torch.Tensor):
        self.lora_gate = lora_gate

    def set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

    def init_lora_param(self):
        for lora_a in self.lora_A_list:
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
        for lora_b in self.lora_B_list:
            nn.init.zeros_(lora_b)

    def merge_lora0_BA_param(self):
        return self.weight.data.transpose(0,1) + \
               (self.lora_A_list[0].transpose(0,1) @ self.lora_B_list[0].transpose(0,1)) * self.scaling

    def forward(self, x: torch.Tensor): 
        assert self.lora_gate is not None
       
        original_output = nn.Linear.forward(self, x) 
        
        if self.r <= 0:
            return original_output
        else:
            if self.training and self.dropout_rate > 0:
                x = self.dropout(x)

            lora_adjustment = x @ self.lora_A_list[self.lora_gate].transpose(0,1) @ self.lora_B_list[self.lora_gate].transpose(0,1)* self.scaling

            return original_output + lora_adjustment

class MultiLoRAFFNLayer(nn.Module):
    def __init__(
            self,
            existing_ffn: nn.Sequential,
            lora_r: int = 0,
            lora_alpha: int = 1,
            dropout_rate: float = 0.,
            num_loras: int = 1,
            params: list = ['mlp.c_fc', 'mlp.c_proj']
            ):
        super().__init__()

        self.c_fc = nn.Linear(existing_ffn.c_fc.in_features, existing_ffn.c_fc.out_features)
        self.c_proj = nn.Linear(existing_ffn.c_proj.in_features, existing_ffn.c_proj.out_features)

        with torch.no_grad():
            self.c_fc.weight.data = existing_ffn.c_fc.weight.data.clone()
            self.c_fc.bias.data = existing_ffn.c_fc.bias.data.clone() if existing_ffn.c_fc.bias is not None else None

            self.c_proj.weight.data = existing_ffn.c_proj.weight.data.clone()
            self.c_proj.bias.data = existing_ffn.c_proj.bias.data.clone() if existing_ffn.c_proj.bias is not None else None

        if 'mlp.c_fc' in params:
            self.c_fc = MultiLoRALinearLayer(
                                     existing_linear=self.c_fc,
                                     lora_r=lora_r,
                                     lora_alpha=lora_alpha,
                                     dropout_rate=dropout_rate,
                                     num_loras=num_loras,
                                     )

        self.gelu = QuickGELU() 

        if 'mlp.c_proj' in params:
            self.c_proj = MultiLoRALinearLayer(
                                       existing_linear = self.c_proj,
                                       lora_r=lora_r,
                                       lora_alpha=lora_alpha,
                                       dropout_rate=dropout_rate,
                                       num_loras=num_loras,
                                       )

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
