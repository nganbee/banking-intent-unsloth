import os
import pandas as pd
from datasets import load_dataset
import re
from sklearn.model_selection import train_test_split

# Library dataset can not use load_dataset(PolyAI/banking77) -> need to use url link from GitHub to download data
TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

DATA_PATH = "../sample_data"

def normalize_text(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():

    train_df = pd.read_csv(TRAIN_DOWNLOAD_URL)
    test_df = pd.read_csv(TEST_DOWNLOAD_URL)
    
    # Rename the category col
    train_df.rename(columns={'category': 'name_intent'}, inplace=True)
    test_df.rename(columns={'category': 'name_intent'}, inplace=True)

    # Label Mapping
    label_names = sorted(train_df['name_intent'].unique().tolist())
    label_to_id = {name: idx for idx, name in enumerate(label_names)}

    train_df['label'] = train_df['name_intent'].map(label_to_id)
    test_df['label'] = test_df['name_intent'].map(label_to_id)
    
    print(f"Length of train: {len(train_df)}")
    print(f"Length of test: {len(test_df)}")
    
    # Applied normalize function for text cols
    train_df['text'] = train_df['text'].apply(normalize_text)
    test_df['text'] = test_df['text'].apply(normalize_text)
    
    # Get 50% of the original data
    sampled_train, _ = train_test_split(
        train_df, 
        train_size=0.5, 
        stratify=train_df['label'], 
        random_state=42
    )
    
    sampled_test, _ = train_test_split(
        test_df, 
        train_size=0.5, 
        stratify=test_df['label'], 
        random_state=42
    )
    
    
    print(f"Length of sampled train: {len(sampled_train)}")
    print(f"Length of sampled test: {len(sampled_test)}")
    
    # Check the directory exist
    os.makedirs(DATA_PATH, exist_ok=True)
    
    # Save to CSV
    train_csv_path = DATA_PATH + '/train.csv'
    test_csv_path =  DATA_PATH + '/test.csv'
    
    sampled_train.to_csv(train_csv_path, index=False)
    sampled_test.to_csv(test_csv_path, index=False)
    
    print(f"Saved to {train_csv_path}")
    print(f"Saved to {test_csv_path}")

if __name__ == "__main__":
    main()