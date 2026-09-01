import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from src.config import PROCESSED_DIR, RANDOM_STATE

FEATURES = ["age", "sex", "localization", "dx_type"]
TARGET = "target"

def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    train_full = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    X_train = train_full[FEATURES]
    y_train = train_full[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    numeric_features = ["age"]
    categorical_features = ["sex", "localization", "dx_type"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))

if __name__ == "__main__":
    main()