import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.preprocessing import LabelEncoder

def _group_titles(titles: pd.Series) -> pd.Series:
    """
    Group rare titles into common categories.

    Args:
        titles: Series containing titles extracted from names.
    Returns:
        Series with grouped titles.
    """

    rare_titles = [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ]

    title_mapping = {
        **{title: "Rare" for title in rare_titles},
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }

    return titles.replace(title_mapping)

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from existing ones.

    Args:
        df: Input DataFrame with raw features.

    Returns:
        DataFrame with engineered features.
    """
    
    df = df.copy()

    # Family-related features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Extract and clean titles
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = _group_titles(df["Title"])


    # Create age bins
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teenager", "Adult", "Middle", "Senior"],
    )

    # Create fare bins
    q = np.quantile(df["Fare"], [0, 0.25, 0.5, 0.75, 1])

    def fare_to_bin(fare):
        if fare <= q[1]:
            return "Low"
        elif fare <= q[2]:
            return "Medium"
        elif fare <= q[3]:
            return "High"
        else:
            return "Very High"
      
    df["FareBin"] = df["Fare"].apply(fare_to_bin)

    return df

def _encode_column(
    train: pd.DataFrame, test: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """
    Encode a single column in both train and test sets.

    Args:
        train: Training DataFrame.
        test: Test DataFrame.
        column: The column name to encode.
    Returns:
        Tuple of (encoded train, encoded test, label encoder).
    """
    le = LabelEncoder()

    # Fit on training data
    train[column] = le.fit_transform(train[column].astype(str))

    # Transform test data, handling unseen categories
    test[column] = (
        test[column]
        .astype(str)
        .apply(lambda x: x if x in le.classes_ else le.classes_[0])
    )
    test[column] = le.transform(test[column])

    return train, test, le

def encode_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical variables using label encoding.

    Args:
        train: Training DataFrame.
        test: Test DataFrame.

    Returns:
        Tuple of (encoded train, encoded test, label encoders dictionary).
    """
    
    train = train.copy()
    test = test.copy()

    categorical_cols = ["Sex", "Embarked", "Title", "AgeBin", "FareBin"]
    label_encoders = {}

    for col in categorical_cols:
        train, test, le = _encode_column(train, test, col)
        label_encoders[col] = le

    return train, test, label_encoders
