import pandas as pd
from sklearn.model_selection import train_test_split

root = r"C:\Users\chand\DeepfakeDataset\celebdf"
labels_path = root + r"\labels.csv"

df = pd.read_csv(labels_path)

# STRATIFIED SPLIT ensures real/fake ratio stays correct
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

train_df.to_csv(root + r"\train.csv", index=False)
val_df.to_csv(root + r"\val.csv", index=False)

print("Train/Val split created.")
print("Train size:", len(train_df))
print("Val size:", len(val_df))

