import pandas as pd
import os

def load_and_aggregate(file_path, user_col='user', header='infer', names=None):
    df = pd.read_csv(file_path, header=header, names=names)
    counts = df[user_col].value_counts().rename(f'{os.path.basename(file_path).replace(".csv","")}_count')
    return counts

def run_feature_engineering_from_files(logon_file, http_file, device_file):
    # logon and device: default headers
    logon_counts = pd.read_csv(logon_file)['user'].value_counts().rename('logon_count') if logon_file else pd.Series(dtype=int)
    device_counts = pd.read_csv(device_file)['user'].value_counts().rename('device_count') if device_file else pd.Series(dtype=int)
    
    # http file: custom columns
    if http_file:
        http_cols = ['id', 'date', 'user', 'pc', 'url']
        http_counts = pd.read_csv(http_file, header=None, names=http_cols)['user'].value_counts().rename('http_count')
    else:
        http_counts = pd.Series(dtype=int)
    
    # merge together
    features = pd.concat([logon_counts, http_counts, device_counts], axis=1).fillna(0).astype(int)
    return features

def main():
    base_dir = os.path.dirname(__file__)
    logon_path = os.path.join(base_dir, '..', 'r1', 'logon.csv')
    http_path = os.path.join(base_dir, '..', 'r1', 'http.csv')
    device_path = os.path.join(base_dir, '..', 'r1', 'device.csv')

    features = run_feature_engineering_from_files(logon_path, http_path, device_path)
    print("Sample of aggregated features:")
    print(features.head())

    features_path = os.path.join(base_dir, '..', 'features.csv')
    features.to_csv(features_path)
    print(f"\nFeatures saved to {features_path}")

if __name__ == "__main__":
    main()
