# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 16:59:49 2025

@author: Mahdi Ghafourian
"""

# build_vocab.py
import os
from utils.ngram_tokenizer import build_balanced_vocab, save_vocab

filepaths = {
    lang.split(".")[0]: os.path.join("data", lang)
    for lang in os.listdir("data") if lang.endswith(".txt")
}


vocab = build_balanced_vocab(
    filepaths=filepaths,
    ngram_range=(2, 4),
    vocab_size=50000,
    max_lines_per_lang=10000
)

save_vocab(vocab, "data/vocab.json")
print("Saved vocabulary to data/vocab.json")

