# Standard library
import argparse

# Third-party library
import langcodes

# PyTorch
import torch

# Project modules
from utils.ngram_tokenizer import load_vocab, encode_ngrams
from utils.extract_text import extract_text_from_file
from utils.model import FastTextClassifier
from utils.dataset import build_label_map

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab and label map
    vocab = load_vocab(args.vocab_path)
    label2id = build_label_map(args.data_dir)
    id2label = {v: k for k, v in label2id.items()}
    
    # Load model
    model = FastTextClassifier(
        vocab_size=len(vocab) + 1,
        embed_dim=args.embed_dim,
        num_classes=len(label2id)
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Extract text from input document
    text = extract_text_from_file(args.input_file)

    # Encode text as n-gram IDs
    token_ids = encode_ngrams(text, vocab, ngram_range=(2, 4))

    if len(token_ids) == 0:
        print("No recognizable n-grams found in input document!")
        return

    text_tensor = torch.tensor(token_ids, dtype=torch.long).to(device)
    offsets = torch.tensor([0], dtype=torch.long).to(device)  # single document → single offset

    with torch.no_grad():
        logits = model(text_tensor, offsets)
        pred_class = logits.argmax(dim=1).item()
        pred_label_code = id2label[pred_class]        
        pred_label = langcodes.Language.get(pred_label_code).language_name()        
        if pred_label == "Unknown language":
            pred_label = pred_label_code

    print(f"Predicted language: {pred_label}")
    # probs = torch.softmax(logits, dim=1)
    # print("Probabilities:", probs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json")
    parser.add_argument("--model_path", type=str, default="models/fasttext_model.pth")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--embed_dim", type=int, default=100)
    args = parser.parse_args()

    predict(args)
