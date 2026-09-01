import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def fit_metadata_encoders(train_csv, val_csv=None):
    train_df = pd.read_csv(train_csv)

    if val_csv is not None:
        val_df = pd.read_csv(val_csv)
        fit_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    else:
        fit_df = train_df.copy()

    fit_df["age"] = fit_df["age"].fillna(fit_df["age"].median())
    fit_df["sex"] = fit_df["sex"].fillna("unknown")
    fit_df["localization"] = fit_df["localization"].fillna("unknown")
    fit_df["dx_type"] = fit_df["dx_type"].fillna("unknown")

    scaler = StandardScaler()
    scaler.fit(fit_df[["age"]])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(fit_df[["sex", "localization", "dx_type"]])

    meta_dim = 1 + encoder.transform(fit_df[["sex", "localization", "dx_type"]].iloc[:1]).shape[1]

    return scaler, encoder, meta_dim


def transform_metadata_row(row, scaler, encoder):
    age = row["age"]
    if pd.isna(age):
        age = 0.0

    age_df = pd.DataFrame([[age]], columns=["age"])
    age_scaled = scaler.transform(age_df)[0]

    cat_df = pd.DataFrame([{
        "sex": row["sex"] if pd.notna(row["sex"]) else "unknown",
        "localization": row["localization"] if pd.notna(row["localization"]) else "unknown",
        "dx_type": row["dx_type"] if pd.notna(row["dx_type"]) else "unknown"
    }])

    cat_encoded = encoder.transform(cat_df)[0]

    metadata_vector = np.concatenate([age_scaled, cat_encoded]).astype(np.float32)
    return metadata_vector