import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
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

BATCH_SIZE = 256
NUM_EPOCHS = 50
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", torch.cuda.device_count(), "GPUs!")
CAT_EMBEDDING_DIM_L = 256
CAT_EMBEDDING_DIM_M = 128
CAT_EMBEDDING_DIM_S = 16
TEXT_EMBEDDING_DIM = 512
HIDDEN_DIM = 512
MAX_HISTORY_LEN = 10
HISTORY_PAD = "[PAD]"
HISTORY_TIME_PAD = 0
TARGET_TYPE = "log"

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_model.to(DEVICE)


def convert_df_to_trainable_set(raw_df):
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
        final_df.append(item)
    return final_df, candidate_embeds


class CustomDataset(Dataset):
    def __init__(self, df, candidate_embeds, target_type="raw"):
        self.df = df
        self.candidate_embeds = candidate_embeds
        self.target_type = target_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        current_sample = self.df[idx]
        if self.target_type == "raw":
            current_sample["label"] = current_sample["cap_days"]
        elif self.target_type == "log":
            current_sample["label"] = np.log1p(current_sample["cap_days"])
        elif self.target_type == "scale":
            current_sample["label"] = current_sample["duration_scaled"]
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
            "cap_days": current_sample["cap_days"],
            "DAYS": current_sample["DAYS"],
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


def asymmetric_mse_loss(preds, gt, overpay_penalty=2.0):
    error = preds - gt
    loss = torch.where(error > 0, overpay_penalty * (error**2), error**2)
    return loss.mean()


def quantile_loss(y_pred, y_true, quantile=0.15):
    error = y_true - y_pred
    return torch.max((quantile - 1) * error, quantile * error).mean()


def combine_loss(y_pred, y_true):
    loss_asy = asymmetric_mse_loss(y_pred, y_true)
    loss_qant = quantile_loss(y_pred, y_true, quantile=0.15)
    # loss_mse = nn.SmoothL1Loss()(y_pred, y_true) # nn.MSELoss()
    return loss_asy * 0.0 + loss_qant * 1


# Define the training loop
def train_model(
    model, train_loader, val_loader, epochs, lr, device, model_save_dir=None
):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = combine_loss

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
                gt += batch["cap_days"].cpu().tolist()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        calc_metrics(np.array(preds), np.array(gt), verbose=True)
        if (epoch + 1) % 5 == 0 and model_save_dir is not None:
            print("Model saveing....")
            torch.save(
                model.state_dict(),  # module.
                f"{model_save_dir}/finetuned_epoch{epoch+1}.pwf",
            )


# Example usage
if __name__ == "__main__":
    CAP_VALUE = 150
    intem_data_save_path = "/app/models/jiaqi/std_din_results"
    # Simulate some random data for training and validation
    df = pd.read_pickle("data_for_experiment.pkl")
    df["cap_days"] = df["DAYS"].apply(lambda x: min(x, CAP_VALUE))
    cat_max_values = df[categorical_cols + boolean_cols].max().tolist()

    CAT_EMBEDDING_DIMS = []
    for x in cat_max_values:
        if x < 10:
            CAT_EMBEDDING_DIMS.append((x + 1, CAT_EMBEDDING_DIM_S))
        elif x < 100:
            CAT_EMBEDDING_DIMS.append((x + 1, CAT_EMBEDDING_DIM_M))
        else:
            CAT_EMBEDDING_DIMS.append((x + 1, CAT_EMBEDDING_DIM_L))
    NUM_FEATURES_DIM = len(numerical_cols)

    scaler = MinMaxScaler()
    df["duration_scaled"] = scaler.fit_transform(df[["cap_days"]])

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
            df[df["data_type"] == "train"]
        )
        final_df_test, candidate_embeds_test = convert_df_to_trainable_set(
            df[df["data_type"] == "test"]
        )
        torch.save(
            (final_df_train, candidate_embeds_train),
            f"{intem_data_save_path}/train.pth",
        )
        torch.save(
            (final_df_test, candidate_embeds_test), f"{intem_data_save_path}/test.pth"
        )

    train_dataset = CustomDataset(
        final_df_train, candidate_embeds_train, target_type=TARGET_TYPE
    )
    val_dataset = CustomDataset(
        final_df_test, candidate_embeds_test, target_type=TARGET_TYPE
    )

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
        din_model,
        train_loader,
        val_loader,
        epochs=NUM_EPOCHS,
        lr=LR,
        device=DEVICE,
        model_save_dir=intem_data_save_path + "/ckpt",
    )
