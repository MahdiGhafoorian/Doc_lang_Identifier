# Standard library
import os
import argparse

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Project modules
from utils.ngram_tokenizer import load_vocab
from utils.dataset import LangIDDataset, collate_batch, build_label_map
from utils.model import FastTextClassifier

def train(args):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load vocab and label map
    vocab = load_vocab(args.vocab_path)
    label2id = build_label_map(args.data_dir)
    # id2label = {v: k for k, v in label2id.items()}
    
    filepaths = {
    lang.split(".")[0]: os.path.join(args.data_dir, lang)
    for lang in os.listdir(args.data_dir)
    if lang.endswith(".txt")
    }
    
    # Create dataset and dataloaders
    full_dataset = LangIDDataset(
        filepaths=filepaths,
        vocab=vocab,
        label2id=label2id,
        ngram_range=(2, 4),
        max_lines_per_lang=args.max_lines_per_lang
    )

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    
    # Initialize model
    model = FastTextClassifier(
        vocab_size=len(vocab) + 1,   # +1 for unknown token
        embed_dim=args.embed_dim,
        num_classes=len(label2id)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        
        for text, offsets, labels in train_loader:
            text, offsets, labels = text.to(device), offsets.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(text, offsets)
            loss = criterion(output, labels)
            loss.backward
            optimizer.step()

            total_train_loss += loss.item()
            
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for text, offsets, labels in val_loader:
                text, offsets, labels = text.to(device), offsets.to(device), labels.to(device)
                outputs = model(text, offsets)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
    
        print(f"Epoch {epoch+1}/{args.epochs} "
              f"Train Loss: {total_train_loss:.4f} | Val Loss: {total_val_loss:.4f}")
        
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "fasttext_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f" Model saved to {model_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json")
    parser.add_argument("--model_dir", type=str, default="models/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_lines_per_lang", type=int, default=10000)
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation set split ratio")
    args = parser.parse_args()

    train(args)
            
            
    
    
    