import pandas as pd
from pathlib import Path
from typing import Literal

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
SUBMISSION_DATA_DIR = BASE_DIR / "data" / "submission"


def _validate_csv_file(file_path: Path) -> None:
    """
    Validate that a CSV file exists and is not empty.
    Args:
        file_path: The path to the CSV file.
    Raises:
    FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or not a CSV.
    """
    if not file_path.parent.exists():
        raise FileNotFoundError(f"The directory {file_path.parent} does not exist.")

    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path.name} does not exist in {file_path.parent}.")

    if file_path.stat().st_size == 0:
        raise ValueError(f"The file {file_path.name} is empty.")


def load_data(
    file_name: str, data_type: Literal["raw", "processed"] = "raw"
) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_name: The name of the CSV file to load.
        data_type: Type of data to load - "raw" or "processed".

    Returns:
        The loaded DataFrame.
    """
    data_dir = RAW_DATA_DIR if data_type == "raw" else PROCESSED_DATA_DIR
    file_path = data_dir / file_name

    _validate_csv_file(file_path)

    df = pd.read_csv(file_path)
    print(f"{file_name} successfully loaded from {file_path}.")

    return df


def load_raw_data(file_name: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.

    Args:
        file_name: The name of the CSV file to load.
    Returns:
        The loaded DataFrame.
    """
    return load_data(file_name, "raw")


def load_processed_data(file_name: str) -> pd.DataFrame:
    """
    Load processed data from a CSV file.

    Args:
        file_name: The name of the CSV file to load.
    Returns:
        The loaded DataFrame.
    """
    return load_data(file_name, "processed")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data by handling missing values and removing duplicates.

    Args:
        df: The input DataFrame.

    Returns:
        The cleaned DataFrame.
    """
    df = df.drop_duplicates()

    # Drop Cabin column (~77% missing values)
    df = df.drop(columns=["Cabin"], errors="ignore")

    # Fill missing values
    imputation_strategy = {
        "Age": df["Age"].median(),
        "Embarked": df["Embarked"].mode().iloc[0],
        "Fare": df["Fare"].median(),
    }

    df = df.fillna(imputation_strategy)

    print("Data cleaning successfully completed.")
    return df


def save_data(
    df: pd.DataFrame,
    file_name: str,
    data_type: Literal["processed", "submission"] = "processed",
) -> None:
    """
    Save data to a CSV file.

    Args:
        df: The DataFrame to save.
        file_name: The name of the CSV file to save.
        data_type: Type of data to save - "processed" or "submission".
    """

    if df.empty:
        raise ValueError("The DataFrame is empty and cannot be saved.")

    data_dir = PROCESSED_DATA_DIR if data_type == "processed" else SUBMISSION_DATA_DIR
    file_path = data_dir / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(file_path, index=False)
    print(f"{file_name} successfully saved to {file_path}")


def save_processed_data(df: pd.DataFrame, file_name: str) -> None:
    """
    Save processed data to a CSV file.

    Args:
        df: The DataFrame to save.
        file_name: The name of the CSV file to save.
    """
    save_data(df, file_name, "processed")


def save_submission_data(df: pd.DataFrame, file_name: str) -> None:
    """
    Save submission data to a CSV file.

    Args:
        df: The DataFrame to save.
        file_name: The name of the CSV file to save.
    """
    save_data(df, file_name, "submission")
