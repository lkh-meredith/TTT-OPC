import os

import torch
import torch.nn as nn

from typing import Dict

from loralib.layers import MultiLoRAFFNLayer

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top1': [11],
        'top2': [10, 11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    else:
        raise NotImplementedError

def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def apply_lora_base(args, clip_model):
    # list_lora_layers = []
    if args.encoder == 'text' or args.encoder == 'both':
        text_encoder = clip_model.transformer
        indices = INDEX_POSITIONS_TEXT[args.position]
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                print(f"Text Residual Attention Block {i}: {block}")
                ffn_lora_experts = MultiLoRAFFNLayer(block.mlp,lora_r=args.lora_r,
                                 lora_alpha=args.lora_alpha,
                                 dropout_rate=args.dropout_rate,
                                 params = args.params)
                setattr(block, "mlp", ffn_lora_experts)

    if args.encoder == 'vision' or args.encoder == 'both':
        vision_encoder = clip_model.visual.transformer
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                print(f"Visual Residual Attention Block {i}: {block}")
                ffn_lora_experts = MultiLoRAFFNLayer(block.mlp, lora_r=args.lora_r,
                                                     lora_alpha=args.lora_alpha,
                                                     dropout_rate=args.dropout_rate,
                                                     params=args.params
                                                     )

                setattr(block, "mlp", ffn_lora_experts)
                

    return

def apply_lora_ttt(args, clip_model):
    if args.encoder == 'text' or args.encoder == 'both':
        text_encoder = clip_model.transformer
        indices = INDEX_POSITIONS_TEXT[args.position]
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                print(f"Text Residual Attention Block {i}: {block}")
                ffn_lora_experts = MultiLoRAFFNLayer(block.mlp,lora_r=args.lora_r,
                                 lora_alpha=args.lora_alpha,
                                 dropout_rate=args.dropout_rate,
                                 num_loras=2, 
                                 params = args.params)

                setattr(block, "mlp", ffn_lora_experts)
              

    if args.encoder == 'vision' or args.encoder == 'both':
        vision_encoder = clip_model.visual.transformer
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                print(f"Visual Residual Attention Block {i}: {block}")
                ffn_lora_experts = MultiLoRAFFNLayer(block.mlp, lora_r=args.lora_r,
                                                     lora_alpha=args.lora_alpha,
                                                     dropout_rate=args.dropout_rate,
                                                     num_loras=2,
                                                     params=args.params
                                                     )
                setattr(block, "mlp", ffn_lora_experts)

    return 
   
def save_lora(args, model, save_lora_path):
    weights = {}
    if args.encoder == "text" or args.encoder == "both":
        indices = INDEX_POSITIONS_TEXT[args.position]
        for i, block in enumerate(model.transformer.resblocks):
            if i in indices:
                if "mlp.c_fc" in args.params:
                    weights[f"model.transformer.resblocks.{i}.mlp.c_fc.lora_A_list"] = block.mlp.c_fc.lora_A_list
                    weights[f"model.transformer.resblocks.{i}.mlp.c_fc.lora_B_list"] = block.mlp.c_fc.lora_B_list
                if "mlp.c_proj" in args.params:
                    weights[f"model.transformer.resblocks.{i}.mlp.c_proj.lora_A_list"] = block.mlp.c_proj.lora_A_list
                    weights[f"model.transformer.resblocks.{i}.mlp.c_proj.lora_B_list"] = block.mlp.c_proj.lora_B_list

    if args.encoder == "vision" or args.encoder == "both":
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        for i, block in enumerate(model.visual.transformer.resblocks):
            if i in indices:
                if "mlp.c_fc" in args.params:
                    weights[f"model.visual.transformer.resblocks.{i}.mlp.c_fc.lora_A_list"] = block.mlp.c_fc.lora_A_list
                    weights[f"model.visual.transformer.resblocks.{i}.mlp.c_fc.lora_B_list"] = block.mlp.c_fc.lora_B_list
                if "mlp.c_proj" in args.params:
                    weights[f"model.visual.transformer.resblocks.{i}.mlp.c_proj.lora_A_list"] = block.mlp.c_proj.lora_A_list
                    weights[f"model.visual.transformer.resblocks.{i}.mlp.c_proj.lora_B_list"] = block.mlp.c_proj.lora_B_list


    metadata = {
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'encoder': args.encoder,
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    torch.save(save_data, save_lora_path)
    print(f'LoRA weights saved to {save_lora_path}')


def load_lora(args, model, save_dir):
    load_path = save_dir+f'/{args.lora_filename}.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['lora_r'] != args.lora_r:
        raise ValueError(
            f"r mismatch: expected {args.lora_r}, found {metadata['lora_r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.lora_r}, found {metadata['lora_r']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")

    weights = loaded_data['weights']
  
    model.load_state_dict(weights)


    print(f'LoRA weights loaded from {load_path}')
    return model

