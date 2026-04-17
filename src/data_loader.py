import pandas as pd
import numpy as np

def load_data(fake_path, true_path):
    print("Loading raw CSV files...")
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    # Assign labels
    fake["label"] = 0
    true["label"] = 1

    # Combine datasets
    data = pd.concat([fake, true], ignore_index=True)

    print(f"Original dataset size: {len(data)}")

    # 1. Remove exact duplicates based on text
    initial_len = len(data)
    data = data.drop_duplicates(subset=['text'])
    print(f"Dropped {initial_len - len(data)} duplicate records.")

    # 2. Remove nulls
    initial_len = len(data)
    data = data.dropna(subset=['text'])
    print(f"Dropped {initial_len - len(data)} null records.")

    # 3. Simple length heuristic (remove < 20 words or > 2000 words)
    # We just split by space for a quick count before full preprocessing
    initial_len = len(data)
    data['word_count'] = data['text'].apply(lambda x: len(str(x).split()))
    
    # Filter bounds
    data = data[(data['word_count'] >= 20) & (data['word_count'] <= 2000)]
    print(f"Dropped {initial_len - len(data)} length outliers (<20 or >2000 words).")

    # Drop the temporary word_count column
    data = data.drop(columns=['word_count'])

    # Random shuffle
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Cleaned dataset size: {len(data)}")

    return data