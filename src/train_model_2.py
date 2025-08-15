# src/train_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

FEATURES_CSV = Path("data/f1_features_clean.csv")
MODEL_OUT = Path("models/f1_winner_model_clean.pkl")

def train_and_evaluate(features_csv: Path, model_out: Path):
    df = pd.read_csv(features_csv, parse_dates=["date"])
    # Ensure season numeric
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    # Temporal split: use last season as validation
    seasons = sorted(df["season"].dropna().unique())
    if len(seasons) >= 2:
        test_season = int(max(seasons))
        train_df = df[df["season"] < test_season].copy()
        test_df = df[df["season"] == test_season].copy()
        print(f"Training seasons: {sorted(set(seasons))[:-1]}  |  Validation season: {test_season}")
    else:
        # fallback: last 20% time-based split
        n = len(df)
        split_idx = int(n * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        print("Single-season dataset: using time-based 80/20 split")

    print("Train rows:", len(train_df), "Val rows:", len(test_df))

    # Build feature matrix: numeric columns except label & 'season'/'round' (we don't want trivial season ID leakage)
    numeric_cols = [c for c in train_df.columns if train_df[c].dtype.kind in "iuf"]
    feature_cols = [c for c in numeric_cols if c not in ("is_winner", "season", "round")]

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["is_winner"].values
    X_val = test_df[feature_cols].fillna(0).values
    y_val = test_df["is_winner"].values

    print("Features used:", feature_cols)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop features:\n", importances.head(10))

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_cols": feature_cols}, model_out)
    print("Saved model to", model_out)

if __name__ == "__main__":
    train_and_evaluate(FEATURES_CSV, MODEL_OUT)