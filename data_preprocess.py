import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

CATEGORICAL_COLS = [
    "PMY_DX_CD10",
    "SEC_DX_CD10",
    "X_TYP_PDR",
    "surg_type",
    "icd_init",
    "icd_init_sec",
    "OCC_TYPE_CD",
    "DBLTY_EVENT_CD",
]
BOOLEAN_COLS = ["SURG_IND", "HOSP_IND", "is_ertw_from_ee"]
NUMERICAL_COLS = [
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
CANDIDATE_COLS = [
    "surg_name",
    "DIAGNOSIS",
    "JOB",
    "SURG_NAME_FROM_ACTIVITIES",
]  # OCC_TYPE_CD can be moved to this candidate, next experiment
HISTORY_COLS = ["ACTIVITIES"]


def reorg_textual_data(df):
    dup_activity_sameday_info = 0
    updated_data = []
    for item in df.to_dict("records"):
        surg_name = item["surg_name"]
        diagnosis = item["DIAGNOSIS"]
        job = item["JOB"]
        surg_name_by_activity = item["SURG_NAME_FROM_ACTIVITIES"]

        candidate_text = f"The surgery name is {surg_name}; the diagnosis is {diagnosis}; the job title is {job}; and the surgery name from activity note is {surg_name_by_activity}."

        activities = item["ACTIVITIES"]
        if not len(activities):
            entry = {
                "timestamp": None,
                "days_ago": 0,
                "CONDITION": None,
                "TREATMENT": None,
                "RL": None,
                "text": "There is no activity notes received.",
            }
            history_output = [entry]
        else:
            # Group data by timestamp
            grouped_data = defaultdict(
                lambda: {"CONDITION": None, "TREATMENT": None, "RL": None}
            )
            for timestamp, value, key in activities:
                if grouped_data[timestamp][key] != value:
                    dup_activity_sameday_info += 1
                grouped_data[timestamp][key] = value

            # Sort timestamps to determine the most recent one
            sorted_timestamps = sorted(grouped_data.keys())
            most_recent_timestamp = sorted_timestamps[-1]  # The latest date

            # Convert to sorted list format
            history_output = []
            for timestamp in sorted_timestamps:
                days_ago = (
                    most_recent_timestamp - timestamp
                ).days  # Calculate days difference
                entry = {
                    "timestamp": timestamp,
                    "days_ago": days_ago,
                    "CONDITION": grouped_data[timestamp]["CONDITION"],
                    "TREATMENT": grouped_data[timestamp]["TREATMENT"],
                    "RL": grouped_data[timestamp]["RL"],
                    "text": f"The condition is {grouped_data[timestamp]['CONDITION']}, "
                    f"treatment is {grouped_data[timestamp]['TREATMENT']}, "
                    f"restriction and limitations is {grouped_data[timestamp]['RL']}.",
                }
                history_output.append(entry)

        updated_data.append(
            {**item, "candidate_text": candidate_text, "history_elems": history_output}
        )
    updated_data = pd.DataFrame(updated_data)
    updated_data = updated_data.drop(columns=CANDIDATE_COLS)
    print(dup_activity_sameday_info)
    return updated_data


def prepare_data_for_training(data, save_path=None):
    target_data = data[
        ["ID", "DAYS"]
        + CATEGORICAL_COLS
        + BOOLEAN_COLS
        + NUMERICAL_COLS
        + CANDIDATE_COLS
        + HISTORY_COLS
    ]
    # fill nall
    target_data.replace({"": "None", None: "None", np.nan: "None"}, inplace=True)
    target_data[NUMERICAL_COLS] = target_data[NUMERICAL_COLS].replace("None", 1e-5)

    # merge text data
    updated_data = reorg_textual_data(target_data)

    # dedup
    updated_data["history_len"] = updated_data["history_elems"].apply(len)
    updated_data["ACTIVITIES"] = updated_data["ACTIVITIES"].astype(str)
    df_dedup = updated_data.drop_duplicates(
        subset=[
            col
            for col in updated_data.columns
            if col != "history_elems" and col != "history_len"
        ]
    )
    df_dedup = df_dedup.drop(columns=["ACTIVITIES"])

    # split train and test by claim ids
    unique_ids = df_dedup["ID"].unique()
    np.random.shuffle(unique_ids)
    split_idx = int(len(unique_ids) * 0.8)
    train_ids, test_ids = unique_ids[:split_idx], unique_ids[split_idx:]
    df_dedup["data_type"] = np.where(df_dedup["ID"].isin(train_ids), "train", "test")

    # transform categorical columns into category int
    df_dedup[CATEGORICAL_COLS + BOOLEAN_COLS] = (
        df_dedup[CATEGORICAL_COLS + BOOLEAN_COLS]
        .astype("category")
        .apply(lambda x: x.cat.codes)
    )

    if save_path is not None:
        df_dedup.to_pickle(f"{save_path}/data_for_experiment.pkl")

    return df_dedup


if __name__ == "__main__":
    with open("/home/b4u20/tmp/transformed_features.p", "rb") as f:
        data = pickle.load(f)
    data = pd.DataFrame(data)
    prepare_data_for_training(data)
