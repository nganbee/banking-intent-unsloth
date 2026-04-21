import os
import argparse
import yaml
import pandas as pd
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, config_path, mode='finetuned'):
        """
        mode: "base_zero_shot, base_few_shot, finetuned
        """
        
        self.config = self._load_config(config_path)
        self.mode = mode
        
        # Check to use base or fine-tune model
        if 'base' in self.mode:
            model_path = self.config['model']['base_model']
        else:
            model_path = self.config['model']['ft_model']
        
        print(f"CREATING {mode.upper()} MODEL")    
        
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
        
    def _get_prompt(self, text):
        few_shot_ex = ""
        if self.mode == "base_few_shot":
             few_shot_ex = """
             ### Examples:
Input: "I can't find my credit card anywhere and I'm worried it's been stolen."
Response: lost_card

Input: "What is the current interest rate for a savings account?"
Response: interest_rate

Input: "I want to transfer 500 dollars to my friend's account."
Response: transfer_funds             
"""
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the intent of the following banking customer query
Output ONLY the specific intent name for this banking query. Do not explain.
{self.class_list_str}
{few_shot_ex}

### Input:
{text}

### Response:
"""
        
    def __call__(self, text):
        
        prompt = self._get_prompt(text)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # Generate output
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.config['model']['max_new_tokens'], 
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.1
        )
        
        # Get the respone
        full_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        prediction = full_text.split("### Response:")[-1].strip()
        return prediction
    
def main():
    parser = argparse.ArgumentParser(description="Banking Intent Evaluation Script")
    
    parser.add_argument("--mode", type=str, default="finetuned", 
                        choices=["base_zero_shot", "base_few_shot", "finetuned"],
                        help="Select mode of model for evalutation")
    
    parser.add_argument("--config", type=str, default="configs/inference.yml", help="Config path")

    args = parser.parse_args()
       
    classifier = IntentClassification(args.config, mode=args.mode)
    
    test_text = "I want to report a stolen card and freeze my account"
    result = classifier(test_text)
    
    print(f"\nTest Input: {test_text}")
    print(f"Model Output: {result}")
    
if __name__ == "__main__":
    main()
    
