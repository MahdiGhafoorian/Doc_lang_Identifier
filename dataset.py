import torch
from torch.utilsls.data import Dataset
from ngram_tokenizer import encode_ngrams


# This class loads training examples (text + language label)
class LangIDDataset(Dataset):
    def __init__(self, filepaths, vocab, label2id, ngram_range=(2, 4), max_lines_per_lang=None):
            """
            Args:
                filepaths (dict): {'en': 'data/en.txt', 'fr': 'data/fr.txt', ...}
                vocab (dict): ngram → ID
                label2id (dict): {'en': 0, 'fr': 1, ...}
                max_lines_per_lang (int or None): Optional cap per language
            """
            self.samples = []
            self.labels = []
            self.vocab = vocab
            self.label2id = label2id
            self.ngram_range = ngram_range
    
            for lang, path in filepaths.items():
                with open(path, encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if max_lines_per_lang and i >= max_lines_per_lang:
                            break
                        line = line.strip()
                        if len(line) == 0:
                            continue
                        self.samples.append(line)
                        self.labels.append(label2id[lang])
                        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        label = self.labels[idx]
        token_ids = encode_ngrams(text, self.vocab, self.ngram_range)
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
def collate_batch(batch):
    """
    Batch a list of (token_ids, label) pairs.

    Returns:
        text_tensor: (sum of lengths)
        offsets_tensor: start index of each sample
        label_tensor: label per sample
    """
    text_list, offsets = [], [0]
    labels = []

    for token_ids, label in batch:
        text_list.append(token_ids)
        labels.append(label)
        offsets.append(offsets[-1] + len(token_ids))

    text_tensor = torch.cat(text_list)
    offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long)
    label_tensor = torch.stack(labels)

    return text_tensor, offsets_tensor, label_tensor
