# Standard library
import os
import argparse

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Project modules
from utils.ngram_tokenizer import load_vocab
from utils.dataset import LangIDDataset, collate_batch, build_label_map
from utils.model import FastTextClassifier

def train(args):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load vocab