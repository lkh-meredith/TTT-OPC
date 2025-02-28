import os
import random
import argparse
import yaml
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils.util_algo import *

from models.lora_clip import LoRACLIP_BASE, DualLoRACLIP_TTT
torch.multiprocessing.set_sharing_strategy('file_system')
from datasets.oxford_pets import OxfordPets
from datasets.eurosat import EuroSAT
from datasets.ucf101 import UCF101
from datasets.sun397 import SUN397
from datasets.caltech101 import Caltech101
from datasets.dtd import DescribableTextures
from datasets.fgvc import FGVCAircraft
from datasets.food101 import Food101
from datasets.oxford_flowers import OxfordFlowers
from datasets.stanford_cars import StanfordCars
from datasets.imagenet import ImageNet
from datasets import build_dataset
from clip import clip

_tokenizer = _Tokenizer()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', default='',
                        help='settings of dataset in yaml format')
    parser.add_argument('--seed', type=int, default=3, metavar='N', help='fix random seed')
    parser.add_argument('--backbone', default='ViT-B/16', type=str)

    parser.add_argument("--lora_r", type=int, default=2, help="Number of prompts to be used in the encoders")
    parser.add_argument('--position', type=str, default='all',
                        choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'],
                        help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['mlp.c_fc'], help='list of attention matrices where putting a LoRA')#'c_proj' 'mlp.c_proj',
    parser.add_argument('--lora_alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate applied before the LoRA module')
   
    parser.add_argument('--alpha',default=0.05,type=float)
    parser.add_argument('--thresh',default=0.6,type=float)
    parser.add_argument('--ratio', default=0.95, type=float)
    parser.add_argument('--ttt', default=True, type=float)
    parser.add_argument('--reg', default=True, type=float)
    parser.add_argument('--consistency', default=True, type=float)
 
    parser.add_argument('--train_epoch', type=int, default=20,
                        help='number of total epochs to run (default: 200)') 
    parser.add_argument('--train_batch_size', default=128, type=int,
                        help='mini-batch size')
    parser.add_argument('--eval_batch_size', default=128, type=int,
                        help='mini-batch size of validation (default: 200)')
    parser.add_argument('--learning_rate_ttt', default=5e-3, type=float,
                        help='learning rate')
    parser.add_argument('--learning_rate_base', default=5e-4, type=float,
                        help='learning rate')
                        
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--print_step', default=5, type=int)
    parser.add_argument('--print_epoch', default=1, type=int)
    parser.add_argument('--weight_decay',default=1e-2, type=float)#for training stage

    parser.add_argument('--save_path', default='./results/')
    args = parser.parse_args()
    return args    

def main():
    # Set Cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare configs and logs
    args = get_arguments()
    set_random_seed(args.seed)
   
    assert(os.path.exists(args.dataset_config))
    cfg = yaml.load(open(args.dataset_config, 'r'), Loader=yaml.Loader)

    logging.basicConfig(level=logging.INFO)

    param_dir = f"{args.save_path}/BASE_OPT/{cfg['dataset']['name']}"
    params = args.params
    print(args.params)
   
    log_file_path = f"{param_dir}/log.txt"
   
    log_directory = os.path.dirname(log_file_path)
    cfg["log"] = {
        "root": log_directory, 
        "model": os.path.join(log_directory, "model"), 
        "prediction": os.path.join(log_directory, "prediction"),
    }
    if not os.path.exists(log_directory): 
        os.makedirs(log_directory)
        os.makedirs(cfg["log"]["model"])
        os.makedirs(cfg["log"]["prediction"])

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.info(args)
    logger.info(cfg)

    # Load clip
    clip_model, transform = clip.load(args.backbone)
    clip_model = clip_model.to(device)

    # Prepare dataset
    logger.info('Preparing dataset')
    base_dataset = build_dataset(dataset=cfg['dataset']['name'], root=cfg['dataset']['root'], subsample='base', num_shots=cfg['dataset']['shots'], transform=transform, type='train', seed=args.seed)
   
    test_dataset_all = build_dataset(dataset=cfg['dataset']['name'], root=cfg['dataset']['root'], subsample='all',num_shots=-1, transform=transform, type='test', seed=args.seed)

    base_loader = DataLoader(dataset=base_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=16)

    test_loader_all = DataLoader(dataset=test_dataset_all, batch_size=args.eval_batch_size,  shuffle=True, num_workers=16,drop_last=False)
   
    logger.info("Training set: {}, Testing set: {}".format(len(base_dataset), len(test_dataset_all)))


    lora_base_path0 = f'{cfg["log"]["model"]}/lora_weights_0.pt'

    if not os.path.exists(lora_base_path0):
        model_base = LoRACLIP_BASE(args=args, cfg=cfg, device=device, clip_model=clip_model)
        model_base.init_lora_list(logger=logger)
        model_base.train(logger=logger, base_loader=base_loader, valid_loader=test_loader_all, save_lora_path=lora_base_path0)
        del model_base, base_loader

    torch.cuda.empty_cache()
    model_ttt = DualLoRACLIP_TTT(args=args, cfg=cfg, device=device, clip_model=clip_model,
                                 base_classnames=base_dataset.classnames)

    model_ttt.init_lora_list(logger=logger, clip_base_lora_path=lora_base_path0)

    lora_base_path1 = f'{cfg["log"]["model"]}/lora_weights_1.pt'
    model_ttt.test_time_tuning(logger=logger, test_loader=test_loader_all,save_lora_path=lora_base_path1)

if __name__ == '__main__':
    main()