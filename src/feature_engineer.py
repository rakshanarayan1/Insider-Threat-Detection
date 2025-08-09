import pandas as pd
import os

def load_and_aggregate(file_path, user_col='user', header='infer', names=None):
    df = pd.read_csv(file_path, header=header, names=names)
    counts = df[user_col].value_counts().rename(f'{os.path.basename(file_path).replace(".csv","")}_count')
    return counts

def main():
    # Base directory
    base_dir = os.path.dirname(__file__)

    # Paths to raw logs (now relative to project root)
    logon_path = os.path.join(base_dir, '..', 'r1', 'logon.csv')
    http_path = os.path.join(base_dir, '..', 'r1', 'http.csv')
    device_path = os.path.join(base_dir, '..', 'r1', 'device.csv')

    # For logon and device files, header and user_col are default
    logon_counts = load_and_aggregate(logon_path)
    device_counts = load_and_aggregate(device_path)

    # For http.csv, load without header and specify columns manually
    http_cols = ['id', 'date', 'user', 'pc', 'url']
    http_counts = load_and_aggregate(http_path, user_col='user', header=None, names=http_cols)

    # Combine all counts into one DataFrame
    features = pd.concat([logon_counts, http_counts, device_counts], axis=1).fillna(0).astype(int)

    print("Sample of aggregated features:")
    print(features.head())

    # Save features CSV in project root
    features_path = os.path.join(base_dir, '..', 'features.csv')
    features.to_csv(features_path)
    print(f"\nFeatures saved to {features_path}")

if __name__ == "__main__":
    main()
