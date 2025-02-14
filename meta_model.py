import torch.nn as nn
import torch


class MetadataCandidateTransformer(nn.Module):
    def __init__(self, cand_dim, meta_dim, hidden_dim=512, num_heads=4, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Linear projections to align metadata and candidate dimensions
        self.cand_proj = nn.Linear(cand_dim, hidden_dim)
        self.meta_proj = nn.Linear(meta_dim, hidden_dim)

        # Cross-attention: Metadata Queries Candidate
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Self-Attention over metadata (to refine after interaction)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Feed-Forward Network (FFN) - Transformer-style
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Final projection back to metadata original dimension
        self.fc_meta = nn.Linear(hidden_dim, meta_dim)

        # Layer Normalization for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Weight initialization
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

    def forward(self, metadata_embedding, candidate_embedding):
        """
        metadata_embedding: (batch_size, meta_dim)
        candidate_embedding: (batch_size, cand_dim)
        """

        # Align dimensions using projection layers
        candidate_embedding = self.cand_proj(candidate_embedding).unsqueeze(
            1
        )  # (batch_size, 1, hidden_dim)

        metadata_embedding = self.meta_proj(metadata_embedding).unsqueeze(
            1
        )  # (batch_size, 1, hidden_dim)

        # --- Cross-Attention (Metadata Queries Candidate) ---
        attn_output, _ = self.cross_attention(
            metadata_embedding, candidate_embedding, candidate_embedding
        )
        attn_output = self.norm1(attn_output + metadata_embedding)  # Skip connection

        # --- Self-Attention over Metadata (Refine after interaction) ---
        self_attn_output, _ = self.self_attention(attn_output, attn_output, attn_output)
        self_attn_output = self.norm2(self_attn_output + attn_output)  # Skip connection

        # --- Feed-Forward Network (FFN) ---
        ffn_output = self.ffn(self_attn_output)
        ffn_output = self.norm3(ffn_output + self_attn_output)  # Skip connection

        # --- Final Projection Back to Original Metadata Dim ---
        result = self.fc_meta(ffn_output).squeeze(1)

        return result


class FeatureCrossNetwork(nn.Module):
    """Feature Cross Network for explicit feature interactions."""

    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.cross_weights = nn.ModuleList(
            [nn.Linear(input_dim, input_dim, bias=True) for _ in range(num_layers)]
        )

    def forward(self, x):
        x0 = x  # Keep original input
        for layer in self.cross_weights:
            x = x0 * layer(x) + x  # Feature interactions
        return x  # Output interaction-enhanced features


class SimpleModel(nn.Module):
    def __init__(
        self,
        cat_embedding_dims,
        num_features_dim,
        num_feature_embed_len=64,
        hidden_dims=128,
    ):
        super().__init__()
        self.num_feature_embed_len = num_feature_embed_len
        self.num_encoder = nn.Sequential(
            nn.Linear(num_features_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims // 2),
            nn.BatchNorm1d(hidden_dims // 2),
            nn.Tanh(),
            nn.Linear(hidden_dims // 2, self.num_feature_embed_len),
            nn.BatchNorm1d(self.num_feature_embed_len),
            nn.Tanh(),
        )

        # Feature Cross Network for interaction learning
        self.feature_cross = FeatureCrossNetwork(num_features_dim)

        # Embedding layers for categorical features
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_size, emb_dim)
                for cat_size, emb_dim in cat_embedding_dims
            ]
        )

    def forward(self, num_features, cat_features):
        num_embeds = self.num_encoder(num_features)
        num_crossed = self.feature_cross(num_features)
        # Encode categorical features
        cat_embeds = [emb(cat_features[i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_embeds = torch.cat(cat_embeds, dim=-1)  # (batch_size, total_cat_dim)

        return torch.cat([num_embeds, num_crossed, cat_embeds], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.BatchNorm1d(input_dim), nn.Tanh()
        )

    def forward(self, x):
        return x + self.fc(x)  # Skip connection


class SimpleModelUpgrade(nn.Module):
    def __init__(
        self,
        cat_embedding_dims,
        num_features_dim,
        num_feature_embed_len=64,
        hidden_dims=512,
    ):
        super().__init__()
        self.num_feature_embed_len = num_feature_embed_len
        self.num_encoder = nn.Sequential(
            nn.Linear(num_features_dim, 128),
            ResidualBlock(128),
            nn.Linear(128, 256),
            ResidualBlock(256),
            nn.Linear(256, hidden_dims),
            ResidualBlock(hidden_dims),
            nn.Linear(hidden_dims, hidden_dims // 2),
            ResidualBlock(hidden_dims // 2),
            nn.Linear(hidden_dims // 2, self.num_feature_embed_len),
            nn.BatchNorm1d(self.num_feature_embed_len),
            nn.Tanh(),
        )

        # Feature Cross Network for interaction learning
        self.feature_cross = FeatureCrossNetwork(num_features_dim, num_layers=2)

        # Embedding layers for categorical features
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_size, emb_dim)
                for cat_size, emb_dim in cat_embedding_dims
            ]
        )

    def forward(self, num_features, cat_features):
        num_embeds = self.num_encoder(num_features)
        num_crossed = self.feature_cross(num_features)

        cat_embeds = [emb(cat_features[i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_embeds = torch.cat(cat_embeds, dim=-1)  # (batch_size, total_cat_dim)

        structured_feat = torch.cat([num_embeds, num_crossed, cat_embeds], dim=-1)

        return structured_feat


class DeepFM(nn.Module):
    def __init__(
        self,
        num_features_dim,
        cat_embedding_dims,
        embedding_dim=16,
        output_dim=128,
        deep_layers=[256, 128],
        dropout=0.2,
    ):
        super().__init__()

        self.num_features_dim = num_features_dim
        self.embedding_dim = embedding_dim

        # 1️⃣ First-Order Linear Term (Bias Term for Each Feature)
        total_cat_size = sum([dim[0] for dim in cat_embedding_dims])
        self.feature_bias = nn.Embedding(
            total_cat_size, 1
        )  # Bias for each categorical feature

        # 2️⃣ Second-Order FM Term (Factorized Interactions)
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(cat_size, embedding_dim)
                for cat_size, _ in cat_embedding_dims
            ]
        )
        self.num_feature_proj = nn.Linear(
            num_features_dim, embedding_dim
        )  # Project numerical features to embedding space

        # 3️⃣ Deep Component (DNN)
        deep_input_dim = (
            embedding_dim + len(cat_embedding_dims) * embedding_dim
        )  # Combined input for DNN

        layers = []
        for layer_size in deep_layers:
            layers.append(nn.Linear(deep_input_dim, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            deep_input_dim = layer_size
        layers.append(nn.Linear(deep_input_dim, output_dim))  # Final regression output
        self.deep_nn = nn.Sequential(*layers)

    def forward(self, num_features, cat_features):
        """
        num_features: (batch_size, num_features_dim)
        cat_features: List of (batch_size,) categorical tensors
        """
        # Compute First-Order Bias Term
        cat_indices = torch.cat(
            [cat_features[i].unsqueeze(1) for i in range(len(cat_features))], dim=1
        )  # (batch_size, num_cat_features)
        first_order_output = self.feature_bias(cat_indices).sum(
            dim=1
        )  # Sum across categorical embeddings

        # Compute Second-Order Factorized Interactions
        cat_embeds = [
            emb(cat_features[i]) for i, emb in enumerate(self.cat_embeddings)
        ]  # List of embeddings
        cat_embeds = torch.cat(
            cat_embeds, dim=1
        )  # (batch_size, num_cat_features * embedding_dim)
        num_embeds = self.num_feature_proj(num_features)  # (batch_size, embedding_dim)
        fm_input = torch.cat(
            [num_embeds, cat_embeds], dim=1
        )  # (batch_size, total_embedding_dim)

        # Compute Deep Component
        deep_output = self.deep_nn(fm_input)

        # Combine All Parts (DeepFM Output)
        final_output = torch.cat([deep_output, first_order_output], dim=1)
        return final_output
