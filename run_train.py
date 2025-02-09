import numpy as np
import pandas as pd
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch.nn as nn
from mobi_din import DINForDurationPrediction, get_bert_embedding
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler


categorical_cols = [
    "PMY_DX_CD10",
    "SEC_DX_CD10",
    "X_TYP_PDR",
    "surg_type",
    "icd_init",
    "icd_init_sec",
    "OCC_TYPE_CD",
    "DBLTY_EVENT_CD",
]
boolean_cols = ["SURG_IND", "HOSP_IND", "is_ertw_from_ee"]
numerical_cols = [
    "mdg_minimum",
    "mdg_optimum",
    "mdg_max",
    "ertw_days",
    "LAST_WORK_DT_DIS_DT",
    "FIRST_TREAT_DT_DIS_DT",
    "FTIME_ANTIC_RTRN_WORK_DT_DIS_DT",
    "FTIME_RTRN_WORK_DT_DIS_DT",
    "PRCDR_DT_DIS_DT",
    "D_BTH_CLN_DIS_DT",
    "rtw_dt",
    "PAID_DURATION",
    "RL_TD",
    "CONDITION_TD",
    "TREATMENT_TD",
]
candidate_cols = ["candidate_text"]
history_cols = ["history_elems"]

BATCH_SIZE = 32
NUM_EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", torch.cuda.device_count(), "GPUs!")
CAT_EMBEDDING_DIM = 8
TEXT_EMBEDDING_DIM = 256
HIDDEN_DIM = 128
MAX_HISTORY_LEN = 10
HISTORY_PAD = "[PAD]"
HISTORY_TIME_PAD = 0
TARGET_TYPE = "log"

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_model.to(DEVICE)


def convert_df_to_trainable_set(raw_df, target_type="raw"):
    candidate_embeds = get_bert_embedding(
        bert_model, bert_tokenizer, raw_df["candidate_text"].tolist()
    )
    final_df = []
    for item in raw_df.to_dict("records"):
        history_elems = item["history_elems"]
        padded_history_elems = [x["text"] for x in history_elems] + [HISTORY_PAD] * (
            MAX_HISTORY_LEN - len(history_elems)
        )
        padded_history_elems = padded_history_elems[:MAX_HISTORY_LEN]
        item["past_text_embeds"] = get_bert_embedding(
            bert_model, bert_tokenizer, padded_history_elems
        )
        padded_past_timestamps = [x["days_ago"] for x in history_elems] + [
            HISTORY_TIME_PAD
        ] * (MAX_HISTORY_LEN - len(history_elems))
        padded_past_timestamps = padded_past_timestamps[:MAX_HISTORY_LEN]
        item["past_timestamps"] = padded_past_timestamps
        if target_type == "raw":
            item["label"] = item["DAYS"]
        elif target_type == "log":
            item["label"] = np.log1p(item["DAYS"])
        elif target_type == "scale":
            item["label"] = item["duration_scaled"]
        final_df.append(item)
    return final_df, candidate_embeds


class CustomDataset(Dataset):
    def __init__(self, df, candidate_embeds):
        self.df = df
        self.candidate_embeds = candidate_embeds

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        current_sample = self.df[idx]
        return {
            "num_features": torch.tensor(
                [current_sample[key] for key in numerical_cols]
            ),
            "cat_features": [
                current_sample[cat] for cat in categorical_cols + boolean_cols
            ],
            "past_text_embeds": current_sample["past_text_embeds"],
            "past_lengths": current_sample["history_len"],
            "past_timestamps": torch.tensor(current_sample["past_timestamps"]).float(),
            "candidate_embed": self.candidate_embeds[idx],
            "targets": torch.tensor(current_sample["label"], dtype=torch.float32),
            "DAYS": current_sample["DAYS"]
        }


def calc_metrics(preds, gt, verbose=False):
    """
    preds and gt can be lists or np arrays
    """
    assert len(preds) == len(gt)
    if TARGET_TYPE == "log":
        preds = np.expm1(preds)
    elif TARGET_TYPE == "scale":
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    avg_approval_days = np.mean(preds)
    # calc overpaid ratio
    overpaid_mask = preds > gt
    num_overpaid_days = sum(overpaid_mask)
    overpaid_ratio = num_overpaid_days / len(preds)
    # calc avg number of overpaid days in the overpaid claims
    approval_diff = preds - gt
    avg_overpaid_days = approval_diff[overpaid_mask].mean()
    if verbose:
        print(f"Average Approved Days: {avg_approval_days}")
        print(f"Overpaid Ratio: {overpaid_ratio}")
        print(
            f"Average Number of Days Overpaid for Overpaid Claims: {avg_overpaid_days}"
        )
    return avg_approval_days, overpaid_ratio, avg_overpaid_days


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
            past_text_embeds = batch["past_text_embeds"].to(device)
            past_lengths = batch["past_lengths"].to(device)
            candidate_embed = batch["candidate_embed"].to(device)
            past_timestamps = batch["past_timestamps"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            outputs = model(
                num_features,
                cat_features,
                past_text_embeds,
                past_lengths,
                candidate_embed,
                past_timestamps,
            )

            # Compute loss
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print("========" * 10)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        gt, preds = [], []
        with torch.no_grad():
            for batch in val_loader:
                num_features = batch["num_features"].to(device)
                cat_features = [cat.to(device) for cat in batch["cat_features"]]
                past_text_embeds = batch["past_text_embeds"].to(device)
                past_lengths = batch["past_lengths"].to(device)
                candidate_embed = batch["candidate_embed"].to(device)
                past_timestamps = batch["past_timestamps"].to(device)
                targets = batch["targets"].to(device)

                outputs = model(
                    num_features,
                    cat_features,
                    past_text_embeds,
                    past_lengths,
                    candidate_embed,
                    past_timestamps,
                )

                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
            preds += outputs.detach().cpu().tolist()
            gt += batch["DAYS"].cpu().tolist()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        calc_metrics(np.array(preds), np.array(gt), verbose=True)


# Example usage
if __name__ == "__main__":
    intem_data_save_path = "/app/models/jiaqi/std_din_results"
    # Simulate some random data for training and validation
    df = pd.read_pickle("data_for_experiment.pkl")
    cat_max_values = df[categorical_cols + boolean_cols].max().tolist()
    CAT_EMBEDDING_DIMS = [(x + 1, CAT_EMBEDDING_DIM) for x in cat_max_values]
    NUM_FEATURES_DIM = len(numerical_cols)

    scaler = MinMaxScaler()
    df["duration_scaled"] = scaler.fit_transform(df[["DAYS"]])

    if os.path.exists(f"{intem_data_save_path}/train.pth"):
        print("============ Loading the saved data ============")
        final_df_train, candidate_embeds_train = torch.load(
            f"{intem_data_save_path}/train.pth"
        )
        final_df_test, candidate_embeds_test = torch.load(
            f"{intem_data_save_path}/test.pth"
        )
    else:
        final_df_train, candidate_embeds_train = convert_df_to_trainable_set(
            df[df["data_type"] == "train"], target_type=TARGET_TYPE
        )
        final_df_test, candidate_embeds_test = convert_df_to_trainable_set(
            df[df["data_type"] == "test"], target_type=TARGET_TYPE
        )
        torch.save(
            (final_df_train, candidate_embeds_train),
            f"{intem_data_save_path}/train.pth",
        )
        torch.save(
            (final_df_test, candidate_embeds_test), f"{intem_data_save_path}/test.pth"
        )

    train_dataset = CustomDataset(final_df_train, candidate_embeds_train)
    val_dataset = CustomDataset(final_df_test, candidate_embeds_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize the model
    din_model = DINForDurationPrediction(
        NUM_FEATURES_DIM,
        CAT_EMBEDDING_DIMS,
        text_in_dim=bert_model.config.hidden_size,
        text_embedding_dim=TEXT_EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        max_seq_len=MAX_HISTORY_LEN,
        dropout_rate=0.2,
    )

    # Train the model
    train_model(
        din_model, train_loader, val_loader, epochs=NUM_EPOCHS, lr=0.001, device=DEVICE
    )
