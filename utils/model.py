import torch
import torch.nn as nn

class FastTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):#, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        # self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.init_weights()
        
    def init_weights(self):
        # nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.zero_()
        
    def forward(self, text, offset):
        """
        Args:
            text (Tensor): [total_tokens] concatenated token IDs of the batch
            offsets (Tensor): [batch_size] where each sequence starts in `text`
        Returns:
            logits (Tensor): [batch_size, num_classes]
        """
        embedded = self.embedding(text, offset) # [batch_size, embed_dim]
        # embedded = self.dropout(embedded) 
        return self.fc(embedded)  # [batch_size, num_classes]
        
        