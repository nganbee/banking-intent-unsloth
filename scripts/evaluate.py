import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from inference import IntentClassification
import re
from difflib import get_close_matches

def normalize_intent_label(text):
    cleaned_text = str(text).strip().lower()
    cleaned_text = re.sub(r"[^a-z0-9]+", "_", cleaned_text)
    cleaned_text = re.sub(r"_+", "_", cleaned_text).strip("_")
    return cleaned_text

def map_to_known_label(prediction, known_labels):
    normalized_prediction = normalize_intent_label(prediction)
    normalized_labels = {normalize_intent_label(label): label for label in known_labels}

    if normalized_prediction in normalized_labels:
        return normalized_labels[normalized_prediction]

    for normalized_label, original_label in normalized_labels.items():
        if normalized_label in normalized_prediction or normalized_prediction in normalized_label:
            return original_label

    close_matches = get_close_matches(normalized_prediction, list(normalized_labels.keys()), n=1, cutoff=0.6)
    if close_matches:
        return normalized_labels[close_matches[0]]

    return normalized_prediction

def main():
    CONFIG_PATH = "../configs/inference.yml"
    classifier = IntentClassification(CONFIG_PATH)

    df_test = pd.read_csv("../sample_data/test.csv")
    known_labels_list = classifier.class_list_str.split("\n") 

    # 4. Chạy dự đoán
    y_true = df_test['name_intent'].tolist()
    y_pred = []

    print(f"Evaluating {len(df_test)} sample test...")

    for i, message in enumerate(tqdm(df_test['text'])):
        raw_pred = classifier(message)
        mapped_pred = map_to_known_label(raw_pred, known_labels_list)
        y_pred.append(mapped_pred)

    print(f"EVALUATION RESULT")
    print("-" * 50)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("-" * 50)
    print(classification_report(y_true, y_pred, digits=4))

    df_test['predicted_intent'] = y_pred
    df_test.to_csv("evaluation_results.csv", index=False)
    
if __name__ == "__main__":
    main()