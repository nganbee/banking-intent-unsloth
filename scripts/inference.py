import os
import yaml
import pandas as pd
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, config_path):
        
        self.config = self._load_config(config_path)
        model_path = self.config['model']['hf_name']
        mapping_path = self.config['data']['map_path']
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = self.config['model']['max_seq_length'],
            load_in_4bit = self.config['model']['load_in_4bit'],
        )
        
        self.class_list_str = self._get_class_intent(mapping_path)
        
        FastLanguageModel.for_inference(self.model)

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def _get_class_intent(self, mapping_path):
        try:
            df_label = pd.read_csv(mapping_path)
            return "\n".join(df_label['name_intent'].tolist())
        
        except Exception as e:
            print(f"Error: {e}")
            return ""

    def __call__(self, text):
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the intent of the following banking customer query
Output ONLY the specific intent name for this banking query. Do not explain.
{self.class_list_str}

### Input:
{text}

### Response:
"""
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # Sinh văn bản (chỉ lấy tối đa 20 tokens cho nhãn)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.config['model']['max_new_tokens'], 
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.1
        )
        
        # Giải mã và cắt lấy phần sau Response:
        full_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        prediction = full_text.split("### Response:")[-1].strip()
        return prediction
    
if __name__ == "__main__":
    classifier = IntentClassification("configs/inference.yml")
    
    test_text = "I want to report a stolen card and freeze my account"
    result = classifier(test_text)
    
    print(f"\nTest Input: {test_text}")
    print(f"Model Output: {result}")
    
