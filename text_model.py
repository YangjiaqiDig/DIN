
import torch.nn as nn
import torch
import torch.nn.functional as F

class TextEmbeddingModule(nn.Module):
    """Encodes free text using BERT."""

    def __init__(self, in_dim, embedding_dim=256):
        super().__init__()
        self.projection = nn.Linear(in_dim, embedding_dim)

    def forward(self, berd_embed):
        return self.projection(berd_embed)  # (batch_size, embedding_dim)


class EnhancedMultiHeadDINAttention(nn.Module):
    """Multi-head attention with enriched query-history interactions and time-aware decay."""

    def __init__(self, input_dim, num_heads=4, time_decay_factor=0.01):
        super().__init__()
        self.num_heads = num_heads
        self.time_decay_factor = (
            time_decay_factor  # Controls how much past interactions decay
        )

        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)

        # Additional layers for interaction terms
        self.interaction_layer = nn.Linear(
            4 * input_dim, input_dim
        )  # To reduce dimensionality

        self.scale_factor = (input_dim // num_heads) ** 0.5

    def forward(
        self, past_interactions, candidate_embedding, past_mask, past_timestamps
    ):
        """
        past_interactions: (batch_size, seq_len, input_dim)
        candidate_embedding: (batch_size, input_dim)
        past_mask: (batch_size, seq_len) - 1 for valid, 0 for padding
        past_timestamps: (batch_size, seq_len) - Time decay factors
        """
        batch_size, seq_len, input_dim = past_interactions.shape

        queries = (
            self.query_layer(candidate_embedding).unsqueeze(1).expand(-1, seq_len, -1)
        )  # (batch_size, seq_len, input_dim)
        keys = self.key_layer(past_interactions)  # (batch_size, seq_len, input_dim)
        values = self.value_layer(past_interactions)  # (batch_size, seq_len, input_dim)

        # Compute interaction features
        interaction_features = torch.cat(
            [
                queries,  # Candidate query
                keys,  # Historical interactions
                queries - keys,  # Element-wise difference
                queries * keys,  # Element-wise product
            ],
            dim=-1,
        )  # (batch_size, seq_len, 4 * input_dim)

        # Reduce interaction feature dimensionality
        interaction_features = self.interaction_layer(
            interaction_features
        )  # (batch_size, seq_len, input_dim)

        # Compute raw attention scores
        attention_scores = (
            torch.bmm(queries, keys.transpose(1, 2)) / self.scale_factor
        )  # (batch_size, seq_len, seq_len)

        # Apply padding mask (set masked positions to -inf before softmax)
        attention_scores = attention_scores.masked_fill(
            past_mask.unsqueeze(1) == 0, float("-inf")
        )

        # Apply time-aware weighting (higher weight for more recent interactions)
        if past_timestamps is not None:
            time_decay = torch.exp(
                -self.time_decay_factor * past_timestamps
            )  # Exponential decay
            attention_scores = attention_scores * time_decay.unsqueeze(1)

        # Compute attention weights
        attention_weights = F.dropout(
            F.softmax(attention_scores, dim=-1), p=0.1
        )  # (batch_size, seq_len, seq_len)

        # Compute weighted sum of values
        attended_features = torch.bmm(
            attention_weights, values
        )  # (batch_size, seq_len, input_dim)
        # attended_features = attended_features.sum(dim=1)
        attended_features = attended_features.mean(dim=1)
        attended_features = (
            attended_features + candidate_embedding
        )  # Residual connection

        # Concatenate attended features with interaction features
        enriched_representation = torch.cat(
            [attended_features, interaction_features.mean(dim=1)], dim=-1
        )  # (batch_size, 2 * input_dim)

        return enriched_representation
