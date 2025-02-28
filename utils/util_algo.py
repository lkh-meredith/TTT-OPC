import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip

def metrics_old(predicts, targets, classnames, base_classnames, all_classnames):
    base_correct, base_total = 0, 0
    sample_selector = [classnames[i] in base_classnames for i in range(len(classnames))]
    class_selector  = [all_classnames[i] in base_classnames for i in range(len(all_classnames))]
    sample_selector = np.array(sample_selector) #2465
    class_selector  = np.array(class_selector) #100
    base_total, new_total = np.sum(sample_selector), np.sum(~sample_selector)
    base_predicts, new_predicts = predicts.copy(), predicts.copy() #2465 100
    base_predicts[:, ~class_selector] = -1e4  #2465 100
    new_predicts[:, class_selector] = -1e4
    base_correct = np.sum(np.argmax(base_predicts[sample_selector, :], axis=1) == targets[sample_selector])
    new_correct = np.sum(np.argmax(new_predicts[~sample_selector, :], axis=1) == targets[~sample_selector])
    total = len(predicts)
    correct = np.sum(np.argmax(predicts, axis=1) == targets)
    return base_correct, base_total, new_correct, new_total, correct, total
