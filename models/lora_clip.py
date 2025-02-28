import os
import time
import yaml
import shutil
import pickle
import random
import argparse
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import torch.nn as nn
import torchvision.transforms as transforms
from loralib.utils import apply_lora_ttt, apply_lora_base, get_lora_parameters, mark_only_lora_as_trainable, save_lora

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import copy
import itertools
from models import TextEncoder
from utils.util_algo import metrics_old, metrics_new
from clip.model import ResidualAttentionBlock
from loralib.layers import MultiLoRAFFNLayer
from collections import defaultdict

_tokenizer = _Tokenizer()

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


class DualLoRACLIP_TTT(object):
    def __init__(self, args, cfg, device, clip_model, base_classnames, K):
    
        self.ttt = args.ttt
        self.reg = args.reg
        self.consistency = args.consistency

        self.model_path = cfg["log"]["model"]
        self.predict_path = cfg["log"]["prediction"]
       
        self.args = args

        self.save_prob = args.save_prob
        self.save_acc = args.save_acc

        self.ratio = args.ratio
        self.thresh = args.thresh

        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        self.params = args.params

        self.alpha = args.alpha
        self.encoder = args.encoder
      
        self.device = device

        self.base_classnames = base_classnames
        # CLIP Model
        self.clip_model = clip_model
       
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
      
        self.templates = cfg["dataset"]["ctx_init"]
        self.indices = INDEX_POSITIONS_VISION[args.backbone][args.position]

        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def init_lora_list(self, logger, clip_base_lora_path=None):
        logger.info(f"========== init lora for test time training ===========")
        apply_lora_ttt(self.args, self.clip_model)
        for k, v in self.clip_model.named_parameters():
            logger.info(k)

        mark_only_lora_as_trainable(self.clip_model)

        if clip_base_lora_path is not None:
            lora_base_dict = torch.load(clip_base_lora_path)["weights"]
           
            if self.encoder == "vision" or self.encoder == "both":
                for idx, block in enumerate(self.clip_model.visual.transformer.resblocks):
                    if idx in self.indices:
                        if "mlp.c_proj" in self.params: 
                            self.clip_model.visual.transformer.resblocks[idx].mlp.c_proj.lora_A_list[0] = \
                                lora_base_dict[f"model.visual.transformer.resblocks.{idx}.mlp.c_proj.lora_A_list"][0].data
                            self.clip_model.visual.transformer.resblocks[idx].mlp.c_proj.lora_B_list[0] = \
                                lora_base_dict[f"model.visual.transformer.resblocks.{idx}.mlp.c_proj.lora_B_list"][0].data
                            logger.info(f"Visual encoder: Load base lora in resblocks.{idx}.mlp.c_proj")

                            self.clip_model.visual.transformer.resblocks[idx].mlp.c_proj.lora_A_list[0].requires_grad = False
                            self.clip_model.visual.transformer.resblocks[idx].mlp.c_proj.lora_B_list[0].requires_grad = False

                        if "mlp.c_fc" in self.params:
                            self.clip_model.visual.transformer.resblocks[idx].mlp.c_fc.lora_A_list[0] = \
                                lora_base_dict[f"model.visual.transformer.resblocks.{idx}.mlp.c_fc.lora_A_list"][0].data
                            self.clip_model.visual.transformer.resblocks[idx].mlp.c_fc.lora_B_list[0] = \
                                lora_base_dict[f"model.visual.transformer.resblocks.{idx}.mlp.c_fc.lora_B_list"][0].data
                            logger.info(f"Visual encoder: Load base lora in resblocks.{idx}.mlp.c_fc")

                            self.clip_model.visual.transformer.resblocks[idx].mlp.c_fc.lora_A_list[0].requires_grad = False
                            self.clip_model.visual.transformer.resblocks[idx].mlp.c_fc.lora_B_list[0].requires_grad = False

            if self.encoder == "text" or self.encoder == "both":
                for idx, block in enumerate(self.clip_model.transformer.resblocks):
                    if idx in self.indices:
                        if "mlp.c_proj" in self.params:
                            self.clip_model.transformer.resblocks[idx].mlp.c_proj.lora_A_list[0] = \
                                lora_base_dict[f"model.transformer.resblocks.{idx}.mlp.c_proj.lora_A_list"][0].data
                            self.clip_model.transformer.resblocks[idx].mlp.c_proj.lora_B_list[0] = \
                                lora_base_dict[f"model.transformer.resblocks.{idx}.mlp.c_proj.lora_B_list"][0].data
                            logger.info(f"Text encoder: Load base lora in resblocks.{idx}.mlp.c_proj")

                            self.clip_model.transformer.resblocks[idx].mlp.c_proj.lora_A_list[0].requires_grad = False
                            self.clip_model.transformer.resblocks[idx].mlp.c_proj.lora_B_list[0].requires_grad = False

                        if "mlp.c_fc" in self.params:
                            self.clip_model.transformer.resblocks[idx].mlp.c_fc.lora_A_list[0] = \
                                lora_base_dict[f"model.transformer.resblocks.{idx}.mlp.c_fc.lora_A_list"][0].data
                            self.clip_model.transformer.resblocks[idx].mlp.c_fc.lora_B_list[0] = \
                                lora_base_dict[f"model.transformer.resblocks.{idx}.mlp.c_fc.lora_B_list"][0].data
                            logger.info(f"Text encoder: Load base lora in resblocks.{idx}.mlp.c_fc")

                            self.clip_model.transformer.resblocks[idx].mlp.c_fc.lora_A_list[0].requires_grad = False
                            self.clip_model.transformer.resblocks[idx].mlp.c_fc.lora_B_list[0].requires_grad = False


        pytorch_total_params = sum(p.numel() for p in self.clip_model.parameters() if p.requires_grad)
        logger.info(f"Updated parameters count: {pytorch_total_params / 1e3:.2f}K")
        if self.ttt:
            logger.info("Test-time training for new lora!")
        else:
            logger.info("Zero-shot for new lora!")

        nouns = torch.load('/home/liukuanghong/DeCoOp/nouns.pt')
        nouns = np.array(nouns)
        # with open('imagenet_class_clean.npy', 'rb') as f:
        #     imagenet_cls = np.load(f) self.base_class
        # percentile = int(0.5*(len(base_classnames)+self.K))
        if not os.path.exists(self.neg_label_index_path):
            percentile = 0.5
            with torch.no_grad():
                texts = [self.templates.format(name.replace("_", " ")) for name in base_classnames]
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  #
                    out_text_tokens = clip.tokenize(texts).to(self.device)
                    text_features = self.clip_model.encode_text(out_text_tokens).to(self.device)  # 未训练之前查找
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    scores = []
                    for word in tqdm(nouns):
                        text = torch.cat([clip.tokenize(f"the " + word)]).to(self.device)
                        sample_features = self.clip_model.encode_text(text).to(self.device)
                        sample_features /= sample_features.norm(dim=-1, keepdim=True)
                        similarity = -(sample_features @ text_features.T)[0]  # 24 #sim=0表示最不相似的数据，-1表示相似的数据
                        # value, index = torch.kthvalue(similarity, percentile)
                        sim = torch.quantile(similarity.float(), q=percentile, dim=-1)
                        scores.append(-sim.cpu().numpy())

            my_list = scores
            sorted_nouns_indices = sorted(range(len(my_list)), key=lambda x: my_list[x])  # 返回排序后的索引
            torch.save(sorted_nouns_indices, self.neg_label_index_path)  # 排序后的索引文件
        else:
            sorted_nouns_indices = torch.load(self.neg_label_index_path)

        # sorted_nouns_indices_group
        # sorted_nouns_indices_topK = sorted_nouns_indices[:self.K // 2] + sorted_nouns_indices[-self.K // 2:]
        negative_label = []
        for i in range(self.neg_group):
            group_negtative_index = [str(nouns[j]) for j in sorted_nouns_indices[i:self.K + i]]
            negative_label.append(group_negtative_index)

        # negative_label = [str(nouns[i]) for i in sorted_nouns_indices[:self.K]]
        if len(negative_label) != 0:
            torch.save(negative_label, self.neg_label_path)
            return negative_label
        else:
            return None

    def test_time_tuning(self, logger, test_loader, save_lora_path):
        self.clip_model = self.clip_model.to(self.device)
    
        if self.ttt:
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.clip_model.parameters()), betas=(0.9, 0.999),
                                               lr=self.args.learning_rate_ttt, weight_decay=0.0) 
           
            self.scaler = torch.cuda.amp.GradScaler(init_scale=1024)#
            torch.autograd.set_detect_anomaly(True)

        self.base_texts = [self.templates.format(name.replace("_", " ")) for name in self.base_classnames]

        all_classnames = test_loader.dataset.classnames
        logger.info(f"all classes num: {len(all_classnames)}")
        self.num_all_classes = len(all_classnames)
        with torch.no_grad():
            texts = [self.templates.format(name.replace("_", " ")) for name in all_classnames]
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16): #
                out_text_tokens = clip.tokenize(texts).to(self.device)

        logger.info("========== Begin Test-time Tuning for OPT ==========")
        last_time = time.time()

        avg_loss = []
        test_cnames = []
        predicts = []
        targets = []
        predicts_ood = []

        for idx, (images, tgt, cnames) in enumerate(tqdm(test_loader)):
            images, tgt = images.to(self.device), tgt.to(self.device) 
            test_cnames += list(cnames)
            targets.append(tgt.cpu())
          
            self.clip_model.eval()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16): 
                with torch.no_grad():
                    if self.encoder == "vision" or self.encoder == "both":
                        for j, block in enumerate(self.clip_model.visual.transformer.resblocks):
                       
                            if j in self.indices:
                                if "mlp.c_fc" in self.params:
                                    block.mlp.c_fc.set_lora_gate(0)
                                if "mlp.c_proj" in self.params:
                                    block.mlp.c_proj.set_lora_gate(0)
                    if self.encoder == "text" or self.encoder == "both":
                        for j, block in enumerate(self.clip_model.transformer.resblocks):
                            
                            if j in self.indices:
                                if "mlp.c_fc" in self.params:
                                    block.mlp.c_fc.set_lora_gate(0)
                                if "mlp.c_proj" in self.params:
                                    block.mlp.c_proj.set_lora_gate(0)

                    image_features_base = self.clip_model.encode_image(images)
                    image_features_base = image_features_base / image_features_base.norm(dim=-1, keepdim=True)
                    logit_scale = self.logit_scale.exp()
                   
                    all_texts_tokens_for_ood = clip.tokenize(self.base_texts).to(self.device)
                    text_features_for_ood = self.clip_model.encode_text(all_texts_tokens_for_ood)
                    text_features_for_ood = text_features_for_ood / text_features_for_ood.norm(dim=-1, keepdim=True)
                    logits_ood_avg = logit_scale * image_features_base @ text_features_for_ood.t()

                    logits_ood_softmax = torch.softmax(logits_ood_avg, dim=-1)
                    MCM_OOD, _ = torch.max(logits_ood_softmax, dim=-1)  
                    ood_detector = MCM_OOD.lt(self.ratio)  
                   
                    predicts_ood.append(logits_ood_softmax)
                   
                    text_features_base = self.clip_model.encode_text(out_text_tokens) 
                    text_features_base = text_features_base / text_features_base.norm(dim=-1, keepdim=True)
                    logits_all = logit_scale * image_features_base @ text_features_base.t()  
                   
                    pseudo_label_all = torch.softmax(logits_all, dim=-1) 
                   
                    MCM_base, label_p_b = torch.max(pseudo_label_all, dim=-1)#128
                    test_mask = MCM_base.lt(self.thresh).float() #
                    base_mask = 1 - test_mask 
                   
                    test_mask = (test_mask.bool() | ood_detector.bool()).float() 
                   
                if self.ttt:
                    self.clip_model.train()
                    with torch.no_grad():
                        if self.encoder == "vision" or self.encoder == "both":
                            for j, block in enumerate(self.clip_model.visual.transformer.resblocks):
                                if j in self.indices:
                                    if "mlp.c_fc" in self.params:
                                        block.mlp.c_fc.set_lora_gate(1)
                                    if "mlp.c_proj" in self.params:
                                        block.mlp.c_proj.set_lora_gate(1)
                        if self.encoder == "text" or self.encoder == "both":
                            for j, block in enumerate(self.clip_model.transformer.resblocks):
                                if j in self.indices:
                                    if "mlp.c_fc" in self.params:
                                        block.mlp.c_fc.set_lora_gate(1)
                                    if "mlp.c_proj" in self.params:
                                        block.mlp.c_proj.set_lora_gate(1)
                  
                        images_test = images.clone()
                        out_text_tokens_test = out_text_tokens.clone()

                    image_features_test = self.clip_model.encode_image(images_test)
                    image_features_test = image_features_test / image_features_test.norm(dim=-1, keepdim=True)

                    text_features_test = self.clip_model.encode_text(out_text_tokens_test)
                    text_features_test = text_features_test / text_features_test.norm(dim=-1, keepdim=True)
                    logits_new = logit_scale * image_features_test @ text_features_test.t()
                    
                    pseudo_label_new = torch.softmax(logits_new, dim=-1)
                    MCM_test, label_p_t = torch.max(pseudo_label_new, dim=-1)  # bs
                   
                    l2_loss_lora = 0.
                    score = 0.
                    loss = 0.
                    if test_mask.sum() != 0:
                        choice_p = MCM_base > MCM_test
                        label_p = torch.where(choice_p, label_p_b, label_p_t)

                        ce_new = (self.cross_entropy(logits_new, label_p) * test_mask).sum() / test_mask.sum()
                        loss = loss + ce_new

                        high_confidence_label_p = (label_p_t == label_p_b) * test_mask 
                        high_quality_pseudo_ratio = high_confidence_label_p.sum() / test_mask.sum() 
                        
                        alpha = self.alpha + (self.alpha*10 - self.alpha) * (1 - high_quality_pseudo_ratio) 
                     
                        logger.info(f"alpha: {alpha}")

                        if self.reg:
                            if self.encoder == "text" or self.encoder == "both":
                                for name, param in self.clip_model.transformer.resblocks.named_parameters():
                                    if "lora_A_list.1" in name:
                                        l2_loss_lora += torch.norm(param, p=2)
                                    if "lora_B_list.1" in name:
                                        l2_loss_lora += torch.norm(param, p=2)

                            if self.encoder == "vision" or self.encoder == "both":
                                for name, param in self.clip_model.visual.transformer.resblocks.named_parameters():
                                    if "lora_A_list.1" in name:
                                        l2_loss_lora += torch.norm(param, p=2)
                                    if "lora_B_list.1" in name:
                                        l2_loss_lora += torch.norm(param, p=2)
                            
                        loss = loss + alpha * l2_loss_lora

                        if self.consistency:
                            if self.encoder == "text" or self.encoder == "both":
                                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)  # Euclidean distance’s 𝐿2-norm
                                score = cos(text_features_base, text_features_test)
                                score = 1.0 - torch.mean(score)
                     
                        loss = loss + alpha * score 

                        avg_loss.append(loss.item())
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                       

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    else:
                        avg_loss.append(loss)
                      
                else:
                    loss = 0.
                    avg_loss.append(loss)
                    self.clip_model.eval()
                    with torch.no_grad():
                        if self.encoder == "vision" or self.encoder == "both":
                            for j, block in enumerate(self.clip_model.visual.transformer.resblocks):
                                if j in self.indices:
                                    if "mlp.c_fc" in self.params:
                                        block.mlp.c_fc.set_lora_gate(1)
                                    if "mlp.c_proj" in self.params:
                                        block.mlp.c_proj.set_lora_gate(1)
                        if self.encoder == "text" or self.encoder == "both":
                            for j, block in enumerate(self.clip_model.transformer.resblocks):
                               
                                if j in self.indices:
                                    if "mlp.c_fc" in self.params:
                                        block.mlp.c_fc.set_lora_gate(1)
                                    if "mlp.c_proj" in self.params:
                                        block.mlp.c_proj.set_lora_gate(1)

                        images_test = images.clone()
                        out_text_tokens_test = out_text_tokens.clone()

                        image_features_test = self.clip_model.encode_image(images_test)
                        image_features_test = image_features_test / image_features_test.norm(dim=-1, keepdim=True)

                        text_features_test = self.clip_model.encode_text(out_text_tokens_test)
                        text_features_test = text_features_test / text_features_test.norm(dim=-1, keepdim=True)
                       
                        logits_new = logit_scale * image_features_test @ text_features_test.t()

                id_select = base_mask.unsqueeze(1).expand(logits_all.shape).bool()
                logits = torch.where(id_select, logits_all, logits_new)

                predicts.append(logits.detach().cpu())

            if (idx + 1) % self.args.print_step == 0 or (idx+1)==len(test_loader):
                base_correct, base_total, new_correct, new_total, correct, total = metrics_old(torch.cat(predicts, 0).numpy(),
                                                                                               torch.cat(targets, 0).numpy(),
                                                                                               test_cnames,
                                                                                               self.base_classnames,
                                                                                               test_loader.dataset.classnames)

                base_acc = base_correct * 100.0 / base_total
                new_acc = new_correct * 100.0 / new_total
                H = 2 * base_acc * new_acc / (base_acc + new_acc)
                acc = correct * 100.0 / total
                logger.info(
                    "Step:[{:3d}/{:3d}]({:.2f}s) Loss:{:.4f} Base:[{:4d}/{:4d}]={:.2f}% New:[{:4d}/{:4d}]={:.2f}% All:[{:4d}/{:4d}]={:.2f}% H: {:.2f}%".format(
                        idx+1, len(test_loader), time.time() - last_time, np.mean(avg_loss),
                        base_correct, base_total, base_correct * 100.0 / base_total,
                        new_correct, new_total, new_correct * 100.0 / new_total,
                        correct, total, acc, H
                    )
                )

                base_correct_new, base_total_new, new_correct_new, new_total_new, _, _ = metrics_new(
                                                                                        torch.cat(predicts, 0).numpy(),
                                                                                        torch.cat(targets, 0).numpy(),
                                                                                        test_cnames,
                                                                                        self.base_classnames,
                                                                                        test_loader.dataset.classnames)
                base_acc_new = base_correct_new * 100.0 / base_total_new
                new_acc_new = new_correct_new * 100.0 / new_total_new
                H_new = 2 * base_acc_new * new_acc_new / (base_acc_new + new_acc_new)

                logger.info(
                    "ComBase:[{:4d}/{:4d}]={:.2f}% ComNew:[{:4d}/{:4d}]={:.2f}% ComH: {:.2f}%".format(
                        base_correct_new, base_total_new, base_acc_new,
                        new_correct_new, new_total_new, new_acc_new,
                        H_new
                    )
                )

                last_time = time.time()

        save_lora(self.args, self.clip_model, save_lora_path)

        logger.info(f"========== Finish! ==========")


class LoRACLIP_BASE(object):
    def __init__(self, args, cfg, device, clip_model):
        self.model_path = cfg["log"]["model"]
        self.predict_path = cfg["log"]["prediction"]
        self.args = args
       
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.params = args.params
        self.device = device
        self.ratio = args.ratio
        self.encoder = args.encoder
        self.clip_model = clip_model

        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
      
        self.templates = cfg["dataset"]["ctx_init"]
        self.indices = INDEX_POSITIONS_VISION[args.backbone][args.position]

        self.cross_entropy = torch.nn.CrossEntropyLoss()


    def init_lora_list(self, logger):
        apply_lora_base(self.args, self.clip_model)
        for k, v in self.clip_model.named_parameters():
            logger.info(k)

        mark_only_lora_as_trainable(self.clip_model)
        pytorch_total_params = sum(p.numel() for p in self.clip_model.parameters() if p.requires_grad)
        logger.info(f"Updated parameters count: {pytorch_total_params / 1e3:.2f}K")


    def train(self, logger, base_loader, valid_loader, save_lora_path):
        self.clip_model = self.clip_model.to(self.device)
        train_loader = base_loader

        with torch.no_grad():
            if self.encoder == "vision" or self.encoder == "both":
                for j, block in enumerate(self.clip_model.visual.transformer.resblocks):
                    if j in self.indices:
                        if "mlp.c_fc" in self.params:
                            block.mlp.c_fc.set_lora_gate(0)
                        if "mlp.c_proj" in self.params:
                            block.mlp.c_proj.set_lora_gate(0)

            if self.encoder == "text" or self.encoder == "both":
                for j, block in enumerate(self.clip_model.transformer.resblocks):
                    if j in self.indices:
                        if "mlp.c_fc" in self.params:
                            block.mlp.c_fc.set_lora_gate(0)
                        if "mlp.c_proj" in self.params:
                            block.mlp.c_proj.set_lora_gate(0)

            self.base_classnames = train_loader.dataset.classnames  # base类别

            self.base_texts = [self.templates.format(name.replace("_", " ")) for name in self.base_classnames]

        self.optimizer = torch.optim.AdamW(get_lora_parameters(self.clip_model), betas=(0.9, 0.999),
                                           lr=self.args.learning_rate_base, weight_decay=self.args.weight_decay) #
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.train_epoch)
        self.scaler = torch.cuda.amp.GradScaler()

        logger.info("========== Begin training ==========")
        last_time = time.time()
        for epoch in range(self.args.train_epoch):
            loss = self.train_epoch(train_loader)

            if (epoch + 1) % self.args.print_epoch == 0 or (epoch + 1) == self.args.train_epoch:
                self.clip_model.eval()
                predicts, targets, classnames, predicts_ood = self.evaluate(logger, valid_loader)

                save_lora(self.args, self.clip_model, save_lora_path)

                base_correct, base_total, new_correct, new_total, correct, total = metrics_old(predicts, targets,
                                                                                               classnames,
                                                                                               train_loader.dataset.classnames,
                                                                                               valid_loader.dataset.classnames)
                base_acc = base_correct * 100.0 / base_total
                new_acc = new_correct * 100.0 / new_total
                H = 2 * base_acc * new_acc / (base_acc + new_acc)
                acc = correct * 100.0 / total
                logger.info(
                    "Step:[{:3d}/{:3d}]({:.2f}s) Loss:{:.4f} Base:[{:4d}/{:4d}]={:.2f}% New:[{:4d}/{:4d}]={:.2f}% All:[{:4d}/{:4d}]={:.2f}% H: {:.2f}%".format(
                        epoch, self.args.train_epoch, time.time() - last_time, loss,
                        base_correct, base_total, base_correct * 100.0 / base_total,
                        new_correct, new_total, new_correct * 100.0 / new_total,
                        correct, total, acc, H
                    )
                )

                last_time = time.time()

        return

    def train_epoch(self, train_loader):
        avg_loss = []
        self.clip_model.train()
        for idx, (images, target, _) in enumerate(tqdm(train_loader)):
            images, target = images.to(self.device), target.to(self.device) 
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  #

                image_features = self.clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                out_text_tokens = clip.tokenize(self.base_texts).to(self.device)
                text_features = self.clip_model.encode_text(out_text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                logits_avg = logit_scale * image_features @ text_features.t() 
              
                loss = self.cross_entropy(logits_avg, target.long())

            self.scaler.scale(loss).backward()

            avg_loss.append(loss.item())

            if ((idx + 1) % self.gradient_accumulation_steps == 0) or (idx + 1 == len(train_loader)):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        self.scheduler.step()

        return np.mean(avg_loss)


    def evaluate(self, logger, loader):
        """This function computes predictions on test data.
        :param data: Dataset object - test dataset
        """
        last_time = time.time()
       
        classnames_eval = loader.dataset.classnames 
        texts_eval = [self.templates.format(name.replace("_", " ")) for name in classnames_eval]


        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  
            out_text_tokens_eval = clip.tokenize(texts_eval).to(self.device)
            predicts, targets, classnames, predicts_ood = [], [], [], []

            with torch.no_grad():
                for idx, (images, labels, cnames) in enumerate(tqdm(loader)):

                    text_features_eval = self.clip_model.encode_text(out_text_tokens_eval) 
                    text_features_eval = text_features_eval / text_features_eval.norm(dim=-1, keepdim=True)

                    images = images.to(self.device) 
                    image_features = self.clip_model.encode_image(images) 
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    logit_scale = self.logit_scale.exp()
                    logits = logit_scale * image_features @ text_features_eval.t() 

                    out_text_tokens_ood = clip.tokenize(self.base_texts).to(self.device)
                    text_features_ood = self.clip_model.encode_text(out_text_tokens_ood)
                    text_features_ood = text_features_ood / text_features_ood.norm(dim=-1, keepdim=True)
                  
                    logits_avg_ood = logit_scale * image_features @ text_features_ood.t()  

                    logits_ood_softmax = torch.softmax(logits_avg_ood, dim=-1)
                    predicts_ood.append(logits_ood_softmax)

                    predicts.append(logits.detach().cpu())
                    targets.append(labels.cpu())
                    classnames += list(cnames)

            predicts = torch.cat(predicts, 0).numpy()
            targets = torch.cat(targets, 0).numpy()
            predicts_ood = torch.cat(predicts_ood, 0)


            logger.info("Evaluate testing set {:.2f}S".format(time.time() - last_time))

            return predicts, targets, classnames, predicts_ood 

