import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class TextEmbeddingModule(nn.Module):
    """ Encodes free text using BERT. """
    def __init__(self, model_name="bert-base-uncased", embedding_dim=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.bert.config.hidden_size, embedding_dim)

    def forward(self, text_input_ids, text_attention_mask):
        outputs = self.bert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        return self.projection(pooled_output)  # (batch_size, embedding_dim)

class MultiHeadDINAttention(nn.Module):
    """ Multi-head attention with time-aware decay for variable-length history. """
    def __init__(self, input_dim, num_heads=4, time_decay_factor=0.01):
        super().__init__()
        self.num_heads = num_heads
        self.time_decay_factor = time_decay_factor  # Controls how much past interactions decay
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)
        self.scale_factor = (input_dim // num_heads) ** 0.5

    def forward(self, past_interactions, candidate_embedding, past_mask, past_timestamps):
        """
        past_interactions: (batch_size, seq_len, input_dim)
        candidate_embedding: (batch_size, input_dim)
        past_mask: (batch_size, seq_len) - 1 for valid, 0 for padding
        past_timestamps: (batch_size, seq_len) - Time decay factors
        """
        queries = self.query_layer(candidate_embedding).unsqueeze(1)  # (batch_size, 1, input_dim)
        keys = self.key_layer(past_interactions)  # (batch_size, seq_len, input_dim)
        values = self.value_layer(past_interactions)  # (batch_size, seq_len, input_dim)

        # Compute raw attention scores
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale_factor  # (batch_size, 1, seq_len)

        # Apply padding mask (set masked positions to -inf before softmax)
        attention_scores = attention_scores.masked_fill(past_mask.unsqueeze(1) == 0, float('-inf'))

        # Apply time-aware weighting (higher weight for more recent interactions)
        if past_timestamps is not None:
            time_decay = torch.exp(-self.time_decay_factor * past_timestamps)  # Exponential decay
            attention_scores = attention_scores * time_decay.unsqueeze(1)

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 1, seq_len)

        # Compute weighted sum of values
        attended_features = torch.bmm(attention_weights, values).squeeze(1)  # (batch_size, input_dim)
        return attended_features
    
class EnhancedMultiHeadDINAttention(nn.Module):
    """ Multi-head attention with enriched query-history interactions and time-aware decay. """
    def __init__(self, input_dim, num_heads=4, time_decay_factor=0.01):
        super().__init__()
        self.num_heads = num_heads
        self.time_decay_factor = time_decay_factor  # Controls how much past interactions decay

        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)

        # Additional layers for interaction terms
        self.interaction_layer = nn.Linear(4 * input_dim, input_dim)  # To reduce dimensionality

        self.scale_factor = (input_dim // num_heads) ** 0.5

    def forward(self, past_interactions, candidate_embedding, past_mask, past_timestamps):
        """
        past_interactions: (batch_size, seq_len, input_dim)
        candidate_embedding: (batch_size, input_dim)
        past_mask: (batch_size, seq_len) - 1 for valid, 0 for padding
        past_timestamps: (batch_size, seq_len) - Time decay factors
        """
        batch_size, seq_len, input_dim = past_interactions.shape

        queries = self.query_layer(candidate_embedding).unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, input_dim)
        keys = self.key_layer(past_interactions)  # (batch_size, seq_len, input_dim)
        values = self.value_layer(past_interactions)  # (batch_size, seq_len, input_dim)

        # Compute interaction features
        interaction_features = torch.cat([
            queries,                # Candidate query
            keys,                   # Historical interactions
            queries - keys,         # Element-wise difference
            queries * keys          # Element-wise product
        ], dim=-1)  # (batch_size, seq_len, 4 * input_dim)

        # Reduce interaction feature dimensionality
        interaction_features = self.interaction_layer(interaction_features)  # (batch_size, seq_len, input_dim)

        # Compute raw attention scores
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale_factor  # (batch_size, seq_len, seq_len)

        # Apply padding mask (set masked positions to -inf before softmax)
        attention_scores = attention_scores.masked_fill(past_mask.unsqueeze(1) == 0, float('-inf'))

        # Apply time-aware weighting (higher weight for more recent interactions)
        if past_timestamps is not None:
            time_decay = torch.exp(-self.time_decay_factor * past_timestamps)  # Exponential decay
            attention_scores = attention_scores * time_decay.unsqueeze(1)

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Compute weighted sum of values
        attended_features = torch.bmm(attention_weights, values)  # (batch_size, seq_len, input_dim)
        attended_features = attended_features.sum(dim=1)
        # attended_features = attended_features.mean(dim=1)  # Average across all queries (batch_size, input_dim)

        # Concatenate attended features with interaction features
        enriched_representation = torch.cat([attended_features, interaction_features.mean(dim=1)], dim=-1)  # (batch_size, 2 * input_dim)


        return enriched_representation

class DINForDurationPrediction(nn.Module):
    """ DIN model for regression with padding + masking for variable-length history. """
    def __init__(self, num_features_dim, cat_embedding_dims, text_embedding_dim=256, hidden_dim=128, max_seq_len=20, dropout_rate=0.2):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        # Embedding layers for categorical features
        self.cat_embeddings = nn.ModuleList([nn.Embedding(cat_size, emb_dim) for cat_size, emb_dim in cat_embedding_dims])
        
        # Text embedding module (BERT-based)
        self.text_embedding = TextEmbeddingModule(embedding_dim=text_embedding_dim)
        
        # Attention for past interactions
        # input_dim = num_features_dim + sum([dim[1] for dim in cat_embedding_dims]) + text_embedding_dim
        input_dim = text_embedding_dim

        # self.attention = MultiHeadDINAttention(input_dim)
        self.attention = EnhancedMultiHeadDINAttention(input_dim)

        # Fully connected layers for regression
        fc_input_dim = num_features_dim + sum([dim[1] for dim in cat_embedding_dims]) + text_embedding_dim * 3  # Concatenated features from all sources
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=True),  # Keep it flexible
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )


    def forward(self, num_features, cat_features, past_text_input_ids, past_text_attention_mask, 
                past_lengths, latest_text_input_ids, latest_text_attention_mask, past_timestamps=None):
        """
        num_features: (batch_size, num_features_dim) - Numerical features
        cat_features: List of (batch_size,) tensors - Categorical features
        past_text_input_ids: (batch_size, seq_len, text_len) - Historical text tokenized
        past_text_attention_mask: (batch_size, seq_len, text_len) - Historical text masks
        past_lengths: (batch_size,) - Actual lengths of each history sequence
        latest_text_input_ids: (batch_size, text_len) - Latest text input (candidate)
        latest_text_attention_mask: (batch_size, text_len) - Latest attention mask
        """

        batch_size, seq_len, text_len = past_text_input_ids.shape
        
        # Encode categorical features
        cat_embeds = [emb(cat_features[i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_embeds = torch.cat(cat_embeds, dim=-1)  # (batch_size, total_cat_dim)

        # Encode latest text (Candidate)
        latest_text_embedding = self.text_embedding(latest_text_input_ids, latest_text_attention_mask)

        # Encode past texts (User History)
        past_text_embeddings = self.text_embedding(past_text_input_ids.view(batch_size * seq_len, text_len),
                                           past_text_attention_mask.view(batch_size * seq_len, text_len))
        past_text_embeddings = past_text_embeddings.view(batch_size, seq_len, -1)  # Reshape back to batch size
        # Compute past mask (Vectorized)
        past_mask = (past_lengths.unsqueeze(-1) > torch.arange(self.max_seq_len, device=past_lengths.device)).float()

        if past_timestamps is not None:
            past_timestamps = past_timestamps * past_mask  # Zero out timestamps for padding positions

        # Compute attention over past interactions
        attended_features = self.attention(past_text_embeddings, latest_text_embedding, past_mask, past_timestamps)


        # Concatenate all features
        x = torch.cat([num_features, cat_embeds, latest_text_embedding, attended_features], dim=-1)

        # Predict duration
        predicted_duration = self.fc(x).squeeze(-1)  # (batch_size,)
        return predicted_duration


if __name__ == "__main__":
    # Test the model
    # Define model parameters
    num_features_dim = 1  # Only one numerical feature
    cat_embedding_dims = [(1000, 8), (500, 8)]  # Two categorical features with embedding dims 8 each
    text_embedding_dim = 256  # Same as BERT output
    hidden_dim = 128
    max_seq_len = 3  # Only 3 historical texts
    dropout_rate = 0.2

    # Initialize the model
    model = DINForDurationPrediction(
        num_features_dim=num_features_dim,
        cat_embedding_dims=cat_embedding_dims,
        text_embedding_dim=text_embedding_dim,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        dropout_rate=dropout_rate
    )

    # Define batch size
    batch_size = 2

    # Create synthetic inputs
    num_features = torch.randn(batch_size, num_features_dim)  # Random numerical feature (e.g., patient age)

    cat_features = [
        torch.randint(0, 1000, (batch_size,)),  # First categorical feature (ICD code)
        torch.randint(0, 500, (batch_size,))    # Second categorical feature (Provider ID)
    ]

    # Simulated tokenized historical text inputs (BERT-style tokenized text)
    seq_len = 3  # 3 history texts
    text_len = 10  # Each text is tokenized into 10 tokens

    past_text_input_ids = torch.randint(0, 30522, (batch_size, seq_len, text_len))  # Random token IDs
    past_text_attention_mask = torch.randint(0, 2, (batch_size, seq_len, text_len))  # Random attention masks

    past_lengths = torch.tensor([3, 2])  # First sample has 3 history texts, second has 2

    # Simulated timestamps (older texts have larger values)
    past_timestamps = torch.randint(1, 100, (batch_size, seq_len)).float()

    # Candidate text (latest free-text entry)
    latest_text_input_ids = torch.randint(0, 30522, (batch_size, text_len))
    latest_text_attention_mask = torch.randint(0, 2, (batch_size, text_len))

    # Run the model
    output = model(
        num_features,
        cat_features,
        past_text_input_ids,
        past_text_attention_mask,
        past_lengths,
        latest_text_input_ids,
        latest_text_attention_mask,
        past_timestamps
    )

    # Check the output
    print("Predicted Duration:", output)
