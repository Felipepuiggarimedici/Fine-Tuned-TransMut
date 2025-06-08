import pandas as pd
import os

print("Current working directory:", os.getcwd())
print("Files in data/trainDataFolds/:", os.listdir('data/trainDataFolds'))

all_dfs = []

for i in range(5):
    train_path = f'data/trainDataFolds/train_data_fold_{i}.csv'
    val_path = f'data/trainDataFolds/val_data_fold_{i}.csv'
    print(f"Trying to load: {train_path} and {val_path}")
    
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
    except FileNotFoundError as e:
        print("ERROR:", e)
        continue

    all_dfs.append(train_df)
    all_dfs.append(val_df)

if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
    merged_df.to_csv('merged_data.csv', index=False)
    print("Merged CSV saved as 'merged_data.csv'")
else:
    print("No dataframes loaded. Check file paths.")
