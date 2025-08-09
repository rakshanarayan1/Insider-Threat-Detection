import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

def main():
    # Get absolute path to features.csv in project root
    base_dir = os.path.dirname(__file__)
    features_path = os.path.join(base_dir, '..', 'features.csv')

    # Load features
    features = pd.read_csv(features_path, index_col=0)

    # Initialize Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

    # Train model on features
    model.fit(features)

    # Predict anomalies: -1 = anomaly, 1 = normal
    preds = model.predict(features)
    features['anomaly'] = preds

    # Save the model in model_store
    model_store_path = os.path.join(base_dir, '..', 'model_store', 'iforest_model.joblib')
    os.makedirs(os.path.dirname(model_store_path), exist_ok=True)
    joblib.dump(model, model_store_path)

    print(f"Model trained and saved to {model_store_path}")
    print("\nSample predictions (anomaly = -1):")
    print(features[features['anomaly'] == -1].head())

if __name__ == "__main__":
    main()
