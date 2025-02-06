import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn as nn
from mobi_din import DINForDurationPrediction

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, num_features, cat_features, past_text_input_ids, past_text_attention_mask, 
                 past_lengths, latest_text_input_ids, latest_text_attention_mask, past_timestamps, targets):
        self.num_features = num_features
        self.cat_features = cat_features
        self.past_text_input_ids = past_text_input_ids
        self.past_text_attention_mask = past_text_attention_mask
        self.past_lengths = past_lengths
        self.latest_text_input_ids = latest_text_input_ids
        self.latest_text_attention_mask = latest_text_attention_mask
        self.past_timestamps = past_timestamps
        self.targets = targets

    def __len__(self):
        return len(self.num_features)

    def __getitem__(self, idx):
        return {
            "num_features": self.num_features[idx],
            "cat_features": [cat[idx] for cat in self.cat_features],
            "past_text_input_ids": self.past_text_input_ids[idx],
            "past_text_attention_mask": self.past_text_attention_mask[idx],
            "past_lengths": self.past_lengths[idx],
            "latest_text_input_ids": self.latest_text_input_ids[idx],
            "latest_text_attention_mask": self.latest_text_attention_mask[idx],
            "past_timestamps": self.past_timestamps[idx],
            "targets": self.targets[idx],
        }

# Define the training loop
def train_model(model, train_loader, val_loader, epochs, lr, device):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # Mean Squared Error for regression

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            # Move data to device
            num_features = batch["num_features"].to(device)
            cat_features = [cat.to(device) for cat in batch["cat_features"]]
            past_text_input_ids = batch["past_text_input_ids"].to(device)
            past_text_attention_mask = batch["past_text_attention_mask"].to(device)
            past_lengths = batch["past_lengths"].to(device)
            latest_text_input_ids = batch["latest_text_input_ids"].to(device)
            latest_text_attention_mask = batch["latest_text_attention_mask"].to(device)
            past_timestamps = batch["past_timestamps"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            outputs = model(
                num_features,
                cat_features,
                past_text_input_ids,
                past_text_attention_mask,
                past_lengths,
                latest_text_input_ids,
                latest_text_attention_mask,
                past_timestamps,
            )

            # Compute loss
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                num_features = batch["num_features"].to(device)
                cat_features = [cat.to(device) for cat in batch["cat_features"]]
                past_text_input_ids = batch["past_text_input_ids"].to(device)
                past_text_attention_mask = batch["past_text_attention_mask"].to(device)
                past_lengths = batch["past_lengths"].to(device)
                latest_text_input_ids = batch["latest_text_input_ids"].to(device)
                latest_text_attention_mask = batch["latest_text_attention_mask"].to(device)
                past_timestamps = batch["past_timestamps"].to(device)
                targets = batch["targets"].to(device)

                outputs = model(
                    num_features,
                    cat_features,
                    past_text_input_ids,
                    past_text_attention_mask,
                    past_lengths,
                    latest_text_input_ids,
                    latest_text_attention_mask,
                    past_timestamps,
                )

                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

# Example usage
if __name__ == "__main__":
    # Simulate some random data for training and validation
    batch_size = 16
    num_samples = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_features = torch.randn(num_samples, 4)  # 4 numerical features
    cat_features = [torch.randint(0, 10, (num_samples,)) for _ in range(2)]  # 2 categorical features
    seq_len = 3
    text_len = 10
    past_text_input_ids = torch.randint(0, 30522, (num_samples, seq_len, text_len))
    past_text_attention_mask = torch.randint(0, 2, (num_samples, seq_len, text_len))
    past_lengths = torch.randint(1, seq_len + 1, (num_samples,))
    latest_text_input_ids = torch.randint(0, 30522, (num_samples, text_len))
    latest_text_attention_mask = torch.randint(0, 2, (num_samples, text_len))
    past_timestamps = torch.randint(1, 100, (num_samples, seq_len)).float()
    targets = torch.randn(num_samples)  # Regression targets

    # Split into train/val sets
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    train_dataset = CustomDataset(
        num_features[:train_size],
        [cat[:train_size] for cat in cat_features],
        past_text_input_ids[:train_size],
        past_text_attention_mask[:train_size],
        past_lengths[:train_size],
        latest_text_input_ids[:train_size],
        latest_text_attention_mask[:train_size],
        past_timestamps[:train_size],
        targets[:train_size],
    )

    val_dataset = CustomDataset(
        num_features[train_size:],
        [cat[train_size:] for cat in cat_features],
        past_text_input_ids[train_size:],
        past_text_attention_mask[train_size:],
        past_lengths[train_size:],
        latest_text_input_ids[train_size:],
        latest_text_attention_mask[train_size:],
        past_timestamps[train_size:],
        targets[train_size:],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the model
    model = DINForDurationPrediction(
        num_features_dim=4,
        cat_embedding_dims=[(10, 8), (10, 8)],  # 2 categorical features with embedding_dim=8
        text_embedding_dim=256,
        hidden_dim=128,
        max_seq_len=seq_len,
        dropout_rate=0.2,
    )

    # Train the model
    train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device=device)
