import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

DATA_FILE = Path("data/f1_features.csv")
MODEL_FILE = Path("models/f1_winner_model.pkl")

def train_model():
    # 1️⃣ Load dataset
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])

    # 2️⃣ Define features and target
    target = "is_winner"
    ignore_cols = ["season", "round", "date", "race_name", "driver_id", "driver_name", "constructor"]
    feature_cols = [c for c in df.columns if c not in ignore_cols + [target]]

    X = df[feature_cols]
    y = df[target]

    # Handle missing values
    X = X.fillna(0)

    # 3️⃣ Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 4️⃣ Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # 5️⃣ Evaluate
    y_pred = model.predict(X_val)
    print("✅ Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    # Show top features
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\n🏎️ Top 10 Features:\n", importances.sort_values(ascending=False).head(10))

    # 6️⃣ Save model
    MODEL_FILE.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, MODEL_FILE)
    print(f"💾 Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_model()