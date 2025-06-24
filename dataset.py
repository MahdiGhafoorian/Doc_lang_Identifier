import torch
from torch.utilsls.data import dataset
from ngram_tokenizer import encode_ngrams


# This class loads training examples (text + language label)
class LangIDDataset(Dataset):
    def __init__: