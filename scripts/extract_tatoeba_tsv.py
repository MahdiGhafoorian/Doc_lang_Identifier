# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 16:11:15 2025

@author: Usuario
"""

# Save this as: scripts/extract_tatoeba_tsv.py

input_path = "../data/spa_sentences.tsv"  # path to .tsv file
output_path = "../data/spa.txt"                 # where cleaned text will go

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:

    for line in infile:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            _, lang_code, sentence = parts
            if lang_code == "spa" and len(sentence.split()) >= 5:
                outfile.write(sentence + "\n")

print(f"Saved cleaned French sentences to: {output_path}")
