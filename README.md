# Document Language Identification

This project implements a **document-level language identification system** using a **FastText-style character n-gram model**.

- Inspired by:
  > Bojanowski et al.,  
  > *Enriching Word Vectors with Subword Information*,  
  > arXiv:1607.04606

It can predict the language of:
- `.txt` files
- `.pdf` documents
- `.docx` documents

## How to Run

1. Prepare training data by putting the in data/ folder

```
data/
├── en.txt   # English lines
├── fr.txt   # French lines
├── de.txt   # German lines
...
```

2. Build the vocabulary by running manually:

```python
python build_vocab.py
```

3. Train the model by running:

```python
python build_vocab.py
```
Or customize:

```python
python train.py \
    --data_dir data/ \
    --vocab_path data/vocab.json \
    --batch_size 64 \
    --epochs 10 \
    --embed_dim 100 \
    --val_split 0.1
```
The trained model will be saved to: 

```
models/fasttext_model.pth
```

4. Predict a test document:

```python
python predict.py --input_file test_documents/sample.pdf
```

## How it works: 

## 1. Input Data Format

For each training sample, the model receives:

1. **Input Text → N-Gram IDs**  
   It is converted into a list of character n-grams. These n-grams are mapped to unique integer IDs using a shared vocabulary.

2. **Embedding Lookup**  
   Each n-gram ID is mapped to a learnable vector (its embedding). For example: 

- `token_ids`: A list of integer IDs representing n-grams  
  **Example:**  
  ```python
  [1023, 88, 1567, 4092]
  ```
- `label`: The correct language ID  
  **Example:**  
  ```python
  2  # (e.g., French)
  ```

---

## 2. Model Architecture (FastText-style)

The model has the following structure:

```python
nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
→ nn.Linear(embed_dim, num_classes)
→ softmax
```

### How it works:

- `EmbeddingBag`: Maps token IDs to vectors and averages them
- `Linear`: Projects the averaged embedding to class scores
- `Softmax`: Converts scores into probabilities

---

## 3. Forward Pass

Given an input like:

```python
token_ids = [103, 87, 421]
label = 1  # (e.g., 'fr')
```

The model:

1. Looks up embeddings:  
   ```python
   [E[103], E[87], E[421]]
   ```
   Each `E[i]` is a learnable vector (e.g., 100-dimensional).

2. Averages the embeddings → one vector:
   ```python
   avg_vector = mean([E[103], E[87], E[421]])
   ```

3. Passes the vector through a linear layer:
   ```python
   scores = Linear(avg_vector)
   ```

4. Applies softmax:
   ```python
   probs = softmax(scores)
   # e.g., [0.1, 0.75, 0.05, 0.1]
   ```

---

## 4. Loss Function: Cross Entropy

- Compares predicted probabilities with the true label.
- Penalizes the model more when the true label has low predicted probability.

Example:
```python
true_label = 1
predicted_probs = [0.1, 0.75, 0.05, 0.1]
loss = CrossEntropy(predicted_probs, true_label)
```

---

## 5. Backpropagation

- Gradients are computed via backpropagation
- Model parameters are updated:
  - Embedding vectors are adjusted (n-grams learn to represent their languages)
  - Linear layer weights are optimized for classification

---

## 6. Repeat Over Thousands of Samples

As training continues with examples like:

```python
[104, 210, 398] → label 1
[93, 11, 276]  → label 2
[401, 312, 122] → label 0
```

The model **learns to associate typical n-gram patterns** with each language class.

---

##  Reference

Bojanowski, P., Grave, E., Joulin, A., Mikolov, T.  
[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606).  
arXiv preprint arXiv:1607.04606 (2016).

---

## License

MIT License





