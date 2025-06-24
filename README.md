# 📚 How the Model Learns (FastText-style)

This model is inspired by the architecture proposed in **fastText** for efficient text classification. It learns to associate character n-gram patterns with language labels through supervised learning.

## Step-by-Step Learning Process: 

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

