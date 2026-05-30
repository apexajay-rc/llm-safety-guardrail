"""
CrisisGuard Dataset Normalizer

Normalizes:

* crisis_type labels
* explicitness labels

Produces:

* clean_echo_dataset.csv
* normalized_stats.json

Author: CrisisGuard Research Team
"""

import json
import logging
from pathlib import Path

import pandas as pd

# ==================================================

# CONFIGURATION

# ==================================================

INPUT_FILE = "echo_dataset.csv"
OUTPUT_FILE = "clean_echo_dataset.csv"
STATS_FILE = "normalized_stats.json"

# ==================================================

# LOGGING

# ==================================================

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(**name**)

# ==================================================

# REQUIRED SCHEMA

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

# NORMALIZATION MAPS

# ==================================================

CRISIS_TYPE_MAP = {

```
# Normal
"normal": "normal",
"none": "normal",

# Stress
"stress": "stress",
"Stress / situational anxiety": "stress",

# Anxiety
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

# Suicide
"suicide": "suicide"
```

}

EXPLICITNESS_MAP = {

```
"explicit": "explicit",

"implicit": "implicit",

# Legacy values
"high": "ambiguous",
"medium": "ambiguous",
"low": "ambiguous"
```

}

# ==================================================

# DATA LOADING

# ==================================================

def load_dataset(path: str) -> pd.DataFrame:

```
logger.info(
    f"Loading dataset: {path}"
)

df = pd.read_csv(path)

# Legacy support
if "user_text" in df.columns:

    logger.info(
        "Detected legacy column "
        "'user_text' -> renaming to 'text'"
    )

    df = df.rename(
        columns={
            "user_text": "text"
        }
    )

return df
```

# ==================================================

# VALIDATION

# ==================================================

def validate_schema(df: pd.DataFrame) -> None:

```
missing_columns = [

    col

    for col in REQUIRED_COLUMNS

    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )

logger.info(
    "Schema validation passed."
)
```

# ==================================================

# NORMALIZATION

# ==================================================

def normalize_crisis_type(
df: pd.DataFrame
) -> pd.DataFrame:

```
df["crisis_type"] = (

    df["crisis_type"]

    .astype(str)

    .str.strip()

    .map(CRISIS_TYPE_MAP)
)

return df
```

def normalize_explicitness(
df: pd.DataFrame
) -> pd.DataFrame:

```
df["explicitness"] = (

    df["explicitness"]

    .astype(str)

    .str.strip()

    .map(EXPLICITNESS_MAP)
)

return df
```

# ==================================================

# QUALITY CHECKS

# ==================================================

def report_unmapped_values(
df: pd.DataFrame
) -> None:

```
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

logger.info(
    f"Unmapped crisis_type rows: "
    f"{missing_crisis_types}"
)

logger.info(
    f"Unmapped explicitness rows: "
    f"{missing_explicitness}"
)
```

def remove_invalid_rows(
df: pd.DataFrame
) -> pd.DataFrame:

```
before = len(df)

df = df.dropna()

after = len(df)

logger.info(
    f"Removed "
    f"{before-after} invalid rows."
)

return df
```

# ==================================================

# STATISTICS

# ==================================================

def generate_statistics(
df: pd.DataFrame
) -> dict:

```
return {

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
```

def save_statistics(
stats: dict,
output_path: str
) -> None:

```
with open(
    output_path,
    "w"
) as f:

    json.dump(
        stats,
        f,
        indent=4
    )

logger.info(
    f"Statistics saved: "
    f"{output_path}"
)
```

# ==================================================

# SAVE DATASET

# ==================================================

def save_dataset(
df: pd.DataFrame,
output_path: str
) -> None:

```
df.to_csv(
    output_path,
    index=False
)

logger.info(
    f"Dataset saved: "
    f"{output_path}"
)
```

# ==================================================

# MAIN

# ==================================================

def main():

```
logger.info(
    "Starting CrisisGuard "
    "dataset normalization."
)

df = load_dataset(
    INPUT_FILE
)

validate_schema(df)

df = normalize_crisis_type(df)

df = normalize_explicitness(df)

report_unmapped_values(df)

df = remove_invalid_rows(df)

stats = generate_statistics(df)

save_dataset(
    df,
    OUTPUT_FILE
)

save_statistics(
    stats,
    STATS_FILE
)

logger.info(
    "Normalization complete."
)
```

if **name** == "**main**":

```
main()
```
