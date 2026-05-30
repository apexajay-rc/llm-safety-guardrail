import pandas as pd
import json


# ==================================================
# CONFIG
# ==================================================

INPUT_FILE = "echo_dataset.csv"

OUTPUT_FILE = "clean_echo_dataset.csv"


# ==================================================
# REQUIRED COLUMNS
# ==================================================

REQUIRED_COLUMNS = [
    "conversation_id",
    "turn_id",
    "text",
    "crisis_level",
    "crisis_type",
    "explicitness"
]


# ==================================================
# CRISIS TYPE NORMALIZATION
# ==================================================

CRISIS_TYPE_MAP = {

    # Normal
    "normal": "normal",
    "none": "normal",

    # Anxiety / Stress
    "stress": "stress",
    "Stress / situational anxiety": "stress",

    "anxiety": "anxiety",

    # Depression
    "depression": "depression",
    "Depression / hopelessness": "depression",
    "hopelessness": "depression",

    # Self Harm
    "self_harm": "self_harm",
    "self_harm_ideation": "self_harm",
    "Self-harm ideation": "self_harm",

    # Suicidal Ideation
    "suicidal_ideation": "suicidal_ideation",
    "Suicidal ideation": "suicidal_ideation",

    # Explicit Suicide
    "suicide": "suicide"
}


# ==================================================
# EXPLICITNESS NORMALIZATION
# ==================================================

EXPLICITNESS_MAP = {

    "explicit": "explicit",

    "implicit": "implicit",

    # Legacy values
    "high": "ambiguous",
    "medium": "ambiguous",
    "low": "ambiguous"
}


# ==================================================
# VALIDATION
# ==================================================

def validate_schema(df):

    missing = []

    for col in REQUIRED_COLUMNS:

        if col not in df.columns:
            missing.append(col)

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("✅ Schema validation passed")


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n================================")
    print("CRISISGUARD DATASET NORMALIZER")
    print("================================")

    # ------------------------------------------
    # Load
    # ------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(
        f"\nLoaded {len(df)} samples"
    )

    print("\nColumns Found:")

    print(df.columns.tolist())

    validate_schema(df)

    # ------------------------------------------
    # Original Stats
    # ------------------------------------------

    print("\nOriginal crisis_type values:")

    print(
        df["crisis_type"]
        .value_counts()
    )

    print("\nOriginal explicitness values:")

    print(
        df["explicitness"]
        .value_counts()
    )

    # ------------------------------------------
    # Normalize crisis_type
    # ------------------------------------------

    df["crisis_type"] = (

        df["crisis_type"]

        .astype(str)

        .str.strip()

        .map(CRISIS_TYPE_MAP)
    )

    # ------------------------------------------
    # Normalize explicitness
    # ------------------------------------------

    df["explicitness"] = (

        df["explicitness"]

        .astype(str)

        .str.strip()

        .map(EXPLICITNESS_MAP)
    )

    # ------------------------------------------
    # Check unmapped rows
    # ------------------------------------------

    missing_crisis_types = (

        df["crisis_type"]

        .isna()

        .sum()
    )

    missing_explicitness = (

        df["explicitness"]

        .isna()

        .sum()
    )

    print("\nValidation Report")
    print("----------------------------")

    print(
        f"Unmapped crisis_type rows: "
        f"{missing_crisis_types}"
    )

    print(
        f"Unmapped explicitness rows: "
        f"{missing_explicitness}"
    )

    # ------------------------------------------
    # Drop invalid rows
    # ------------------------------------------

    before = len(df)

    df = df.dropna()

    after = len(df)

    print(
        f"Dropped {before-after} invalid rows"
    )

    # ------------------------------------------
    # Final Stats
    # ------------------------------------------

    print("\nNormalized crisis_type values")

    print(
        df["crisis_type"]
        .value_counts()
    )

    print("\nNormalized explicitness values")

    print(
        df["explicitness"]
        .value_counts()
    )

    # ------------------------------------------
    # Save Clean Dataset
    # ------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ------------------------------------------
    # Save Stats
    # ------------------------------------------

    stats = {

        "total_samples":
            int(len(df)),

        "crisis_type_distribution":

            df["crisis_type"]
            .value_counts()
            .to_dict(),

        "explicitness_distribution":

            df["explicitness"]
            .value_counts()
            .to_dict()
    }

    with open(
        "normalized_stats.json",
        "w"
    ) as f:

        json.dump(
            stats,
            f,
            indent=4
        )

    print(
        f"\n✅ Saved: {OUTPUT_FILE}"
    )

    print(
        "✅ Saved: normalized_stats.json"
    )

    print(
        "\nNormalization Complete"
    )


# ==================================================
# ENTRYPOINT
# ==================================================

if __name__ == "__main__":

    main()
