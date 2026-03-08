import pandas as pd

def load_data(fake_path, true_path):
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true])
    data = data.sample(frac=1).reset_index(drop=True)

    return data