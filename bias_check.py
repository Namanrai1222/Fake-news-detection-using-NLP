from src.data_loader import load_data
from src.config import FAKE_DATA_PATH, TRUE_DATA_PATH
import matplotlib.pyplot as plt
import re
data = load_data(FAKE_DATA_PATH, TRUE_DATA_PATH)

data["length"] = data["text"].apply(len)

fake_lengths = data[data["label"] == 0]["length"]
real_lengths = data[data["label"] == 1]["length"]

print("Fake avg length:", fake_lengths.mean())
print("Real avg length:", real_lengths.mean())

plt.figure()
plt.hist(fake_lengths, bins=50, alpha=0.5, label="Fake")
plt.hist(real_lengths, bins=50, alpha=0.5, label="Real")
plt.legend()
plt.title("Article Length Distribution")
plt.show()
print("\nSubject distribution by class:\n")
print(data.groupby("label")["subject"].value_counts())
reuters_mask = data["text"].str.contains("Reuters", case=False, na=False)
reuters_count = reuters_mask.sum()

print("\nArticles containing 'Reuters':", reuters_count)

print("\nLabel distribution for Reuters articles:")
print(data[reuters_mask]["label"].value_counts())