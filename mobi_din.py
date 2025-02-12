import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from meta_model import MetadataCandidateTransformer, DeepFM, SimpleModel
from text_model import TextEmbeddingModule, EnhancedMultiHeadDINAttention


def get_bert_embedding(
    bert_model, bert_tokenizer, text_list, batch_size=125, device="cuda"
):
    embeddings = []
    for i in range(0, len(text_list), batch_size):  # Process in batches of 256
        batch_texts = text_list[i : i + batch_size]
        tokens = bert_tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = bert_model(**tokens)
        batch_embeddings = outputs.last_hidden_state[
            :, 0, :
        ].cpu()  # Extract [CLS] token embedding
        embeddings.append(batch_embeddings)
        del tokens, outputs, batch_embeddings
        torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)  # Concatenate batch results


class DINForDurationPrediction(nn.Module):
    """DIN model for regression with padding + masking for variable-length history."""

    def __init__(
        self,
        num_features_dim,
        cat_embedding_dims,
        text_in_dim,
        text_embedding_dim=256,
        hidden_dim=128,
        max_seq_len=20,
        dropout_rate=0.2,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.num_feature_embed_len = 32

        self.metaModel = SimpleModel(cat_embedding_dims, num_features_dim, self.num_feature_embed_len)
        # self.metaModel = DeepFM(num_features_dim, cat_embedding_dims, embedding_dim=16, deep_layers=[128, 64])

        # Text embedding module (BERT-based)
        self.text_embedding = TextEmbeddingModule(
            text_in_dim, embedding_dim=text_embedding_dim
        )

        # Attention for past interactions
        self.attention = EnhancedMultiHeadDINAttention(text_embedding_dim)
        meta_dim = self.num_feature_embed_len + sum(
            [dim[1] for dim in cat_embedding_dims]
        ) + 15
        self.metadata_candidate_attention = MetadataCandidateTransformer(
            cand_dim=text_embedding_dim, meta_dim=meta_dim
        )

        # Fully connected layers for regression
        fc_input_dim = (
            meta_dim + text_embedding_dim * 2
        )  # Concatenated features from all sources

        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

    def forward(
        self,
        num_features,
        cat_features,
        past_text_embeds,
        past_lengths,
        candidate_embed,
        past_timestamps=None,
    ):
        """
        num_features: (batch_size, num_features_dim) - Numerical features
        cat_features: List of (batch_size,) tensors - Categorical features
        past_text_embeds: (batch_size, seq_len, text_len) - Historical text tokenized
        past_lengths: (batch_size,) - Actual lengths of each history sequence
        candidate_embed: (batch_size, text_len) - Latest text input (candidate)
        """

        batch_size, seq_len, text_len = past_text_embeds.shape

        # Encode latest text (Candidate)
        latest_text_embedding = self.text_embedding(candidate_embed)
        
        # Encode past texts (User History)
        past_text_embeddings = self.text_embedding(
            past_text_embeds.view(batch_size * seq_len, text_len)
        ).view(
            batch_size, seq_len, -1
        )  # Reshape back to batch size

        # Compute past mask (Vectorized)
        past_mask = (
            past_lengths.unsqueeze(-1)
            > torch.arange(self.max_seq_len, device=past_lengths.device)
        ).float()

        if past_timestamps is not None:
            # Zero out timestamps for padding positions
            past_timestamps = past_timestamps * past_mask

        # Compute attention over past interactions
        text_attended_features = self.attention(
            past_text_embeddings, latest_text_embedding, past_mask, past_timestamps
        )

        metadata_embedding = self.metaModel(num_features, cat_features)
        # metadata_embedding = torch.cat([num_embeds, cat_embeds], dim=-1)
        metadata_candidate_interaction = self.metadata_candidate_attention(
            metadata_embedding, latest_text_embedding
        )

        # Concatenate all features
        x = torch.cat(
            [
                metadata_candidate_interaction,
                text_attended_features,
            ],
            dim=-1,
        )

        # Predict duration
        predicted_duration = self.fc(x).squeeze(-1)  # (batch_size,)
        return predicted_duration


if __name__ == "__main__":
    # Load BERT model and tokenizer
    model_name = "bert-base-uncased"
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    bert_model.to("cuda")

    # Test the model
    # Define model parameters
    num_features_dim = 10
    cat_embedding_dims = [
        (1000, 8),
        (500, 8),
    ]  # Two categorical features with embedding dims 8 each
    text_embedding_dim = 256
    hidden_dim = 128
    max_seq_len = 3  # Only 3 historical texts
    dropout_rate = 0.2

    # Initialize the model
    model = DINForDurationPrediction(
        num_features_dim=num_features_dim,
        cat_embedding_dims=cat_embedding_dims,
        text_in_dim=bert_model.config.hidden_size,
        text_embedding_dim=text_embedding_dim,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        dropout_rate=dropout_rate,
    )

    # Define batch size
    batch_size = 2

    # Create synthetic inputs
    num_features = torch.randn(
        batch_size, num_features_dim
    )  # Random numerical feature (e.g., patient age)

    cat_features = [
        torch.randint(0, 1000, (batch_size,)),  # First categorical feature (ICD code)
        torch.randint(
            0, 500, (batch_size,)
        ),  # Second categorical feature (Provider ID)
    ]

    # Simulated tokenized historical text inputs (BERT-style tokenized text)
    seq_len = 3  # 3 history texts
    past_texts = [
        ["hello my name is Jackie", "Good morning, how are you?", "I'm doing good"],
        [
            "Simulated tokenized historical text inputs",
            "Lets create random samples",
            "padding",
        ],
    ]
    past_text_embeds = torch.stack(
        [get_bert_embedding(bert_model, bert_tokenizer, x) for x in past_texts], dim=0
    )
    past_lengths = torch.tensor(
        [3, 2]
    )  # First sample has 3 history texts, second has 2

    # Simulated timestamps (older texts have larger values)
    past_timestamps = torch.randint(1, 100, (batch_size, seq_len)).float()

    # Candidate text (latest free-text entry)
    latest_texts = ["hey there, lets ge predict-1!", "hey there, lets ge predict-2!"]
    candidate_embed = get_bert_embedding(bert_model, bert_tokenizer, latest_texts)

    # Run the model
    output = model(
        num_features,
        cat_features,
        past_text_embeds,
        past_lengths,
        candidate_embed,
        past_timestamps,
    )

    # Check the output
    print("Predicted Duration:", output)
