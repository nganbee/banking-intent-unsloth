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
        
        # For predict_batch
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
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
             few_shot_ex = """### Examples:
Input: "I can't find my credit card anywhere and I'm worried it's been stolen."
Response: lost_card

Input: "What is the current interest rate for a savings account?"
Response: interest_rate

Input: "I want to transfer 500 dollars to my friend's account."
Response: transfer_funds             
"""
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the intent of the following banking customer query.
Rule: Output ONLY the exact intent name in these label.
{self.class_list_str}
{few_shot_ex}

### Input:
{text}

### Response:
"""

    def _predict_batch(self, texts):
        prompts = [self._get_prompt(text) for text in texts]
        
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, # Tự động thêm padding cho các câu ngắn hơn
            truncation=True,
            max_length=self.config['model']['max_seq_length']
        ).to("cuda")
        
        # 3. Generate hàng loạt
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.config['model']['max_new_tokens'],
            temperature=0,
            use_cache=True
        )
        
        # 4. Decode và tách kết quả
        full_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = [text.split("### Response:")[-1].strip() for text in full_texts]
        
        return predictions
        
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
            temperature=0
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
    parser.add_argument("--interactive", action="store_true", help="Input text to model directly")

    args = parser.parse_args()
    classifier = IntentClassification(args.config, mode=args.mode)
    
    if args.interactive:
        print("\n" + "="*50)
        print(f"START TYPING ({args.mode.upper()})")
        print("Type 'exit' or 'q' to stop")
        print("="*50)
        
        while True:
            user_input = input("\nInput text: ").strip()
            
            if user_input.lower() in ['exit', 'q', 'quit']:
                print("Quiting...")
                break
                
            if not user_input:
                continue
                
            result = classifier(user_input)
            
            print(f"Model ouptut: {result}")
    else:
        test_text = "How do I reset my secret code? I think I forgot it."
        result = classifier(test_text)
        
        print(f"\nTest Input: {test_text}")
        print(f"Model Output: {result}")
    
if __name__ == "__main__":
    main()
    
