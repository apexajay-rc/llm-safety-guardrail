"""
CrisisGuard Dataset Builder

Version 1:
- Loads Echo dataset
- Validates schema
- Removes duplicates
- Generates statistics
- Creates train/val/test splits

Author: CrisisGuard Research Team
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

RAW_DATA = "data/raw/echo_dataset.csv"

OUTPUT_DIR = "data/processed"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# --------------------------------------------------
# REQUIRED COLUMNS
# --------------------------------------------------

REQUIRED_COLUMNS = [

    "text",

    "crisis_level",

    "crisis_type",

    "explicitness",

    "conversation_id",

    "turn_id"
]


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate_schema(df):

    missing = []

    for col in REQUIRED_COLUMNS:

        if col not in df.columns:
            missing.append(col)

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    print("✅ Schema validation passed")


# --------------------------------------------------
# DATASET STATS
# --------------------------------------------------

def generate_stats(df):

    stats = {

        "total_samples": len(df),

        "unique_conversations":
            df["conversation_id"]
            .nunique(),

        "crisis_levels":
            df["crisis_level"]
            .value_counts()
            .sort_index()
            .to_dict(),

        "crisis_types":
            df["crisis_type"]
            .value_counts()
            .to_dict(),

        "explicitness":
            df["explicitness"]
            .value_counts()
            .to_dict()
    }

    return stats


# --------------------------------------------------
# SPLIT DATASET
# --------------------------------------------------

def split_dataset(df):

    train_df, temp_df = train_test_split(

        df,

        test_size=0.2,

        random_state=42,

        stratify=df["crisis_level"]
    )

    val_df, test_df = train_test_split(

        temp_df,

        test_size=0.5,

        random_state=42,

        stratify=temp_df["crisis_level"]
    )

    return train_df, val_df, test_df


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\nLoading dataset...")

    df = pd.read_csv(RAW_DATA)

    print(
        f"Loaded {len(df)} samples"
    )

    validate_schema(df)

    # ----------------------------------
    # Remove duplicate texts
    # ----------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["text"]
    )

    after = len(df)

    print(
        f"Removed {before-after} duplicates"
    )

    # ----------------------------------
    # Stats
    # ----------------------------------

    stats = generate_stats(df)

    Path(
        OUTPUT_DIR
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    with open(

        f"{OUTPUT_DIR}/dataset_stats.json",

        "w"

    ) as f:

        json.dump(
            stats,
            f,
            indent=4
        )

    print(
        "✅ Stats saved"
    )

    # ----------------------------------
    # Split
    # ----------------------------------

    train_df, val_df, test_df = (
        split_dataset(df)
    )

    train_df.to_csv(
        f"{OUTPUT_DIR}/train.csv",
        index=False
    )

    val_df.to_csv(
        f"{OUTPUT_DIR}/val.csv",
        index=False
    )

    test_df.to_csv(
        f"{OUTPUT_DIR}/test.csv",
        index=False
    )

    print(
        f"Train: {len(train_df)}"
    )

    print(
        f"Validation: {len(val_df)}"
    )

    print(
        f"Test: {len(test_df)}"
    )

    print(
        "\n✅ Dataset Builder Complete"
    )


if __name__ == "__main__":

    main()
