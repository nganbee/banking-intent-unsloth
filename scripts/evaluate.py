import argparse
import sys
import gc
import torch
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

    return "unknown"

def evaluate_model(mode, config_path):
    classifier = IntentClassification(mode, config_path)
    
    df_test = pd.read_csv(classifier.config['data']['test_path'])
    known_labels_list = classifier.class_list_str.split("\n")
    
    y_true = df_test['name_intent'].tolist()
    y_pred = []
    
    print(f"Evaluating model {mode.upper()}")
    
    for _, message in enumerate(tqdm(df_test['text'])):
        raw_pred = classifier(message)
        mapped_pred = map_to_known_label(raw_pred, known_labels_list)
        y_pred.append(mapped_pred)
        
    print(f"EVALUATION RESULT")
    print("-" * 50)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("-" * 50)
    print(classification_report(y_true, y_pred, digits=4))
    
    df_test['predicted_intent'] = y_pred
    df_test.to_csv(f"evaluation_results_{mode}.csv", index=False)
    
    # Clean VRAM
    del classifier
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Clean VRAM {mode}\n")
    
    return y_pred
    
def main():
    CONFIG_PATH = "configs/inference.yml"
    
    parser = argparse.ArgumentParser(description="Banking Intent Evaluation Script")
    
    parser.add_argument("--mode", type=str, default="finetuned", 
                        choices=["base_zero_shot", "base_few_shot", "finetuned", "all"],
                        help="Select mode of model for evalutation")
    
    parser.add_argument("--config", type=str, default="configs/inference.yml", help="Config path")

    args = parser.parse_args()
    if args.mode == "all":
        modes_to_run = ["base_zero_shot", "base_few_shot", "finetuned"]
        print("Evaluation for all model")
    else:
        modes_to_run = [args.mode]

    for current_mode in modes_to_run:
        try:
            evaluate_model(mode=current_mode, config_path=args.config)
        except Exception as e:
            print(f"Error while running {current_mode}: {e}")
            sys.exit(1)
    
if __name__ == "__main__":
    main()