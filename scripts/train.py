from unsloth import FastLanguageModel
import yaml
import os

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset, Dataset
import torch
import pandas as pd

CONFIG_PATH = '/kaggle/input/datasets/kimngntrn510/bank-intent-data/sample_data/configs/train.yml'
TRAIN_PATH = '/kaggle/input/datasets/kimngntrn510/bank-intent-data/sample_data/sample_data/train.csv'

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the intent of the following banking customer query.
{}

### Input:
{}

### Response:
{}"""

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("Load config")
    config = load_config(CONFIG_PATH)
    
    print("Loading model")
    # Prepare model, tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config['model']['model_name'],
        max_seq_length = config['model']['max_seq_length'],
        dtype = None,
        load_in_4bit = config['model']['load_in_4bit'],
    )

    print("Loading LoRA Adapter")
    # LoRA Configs
    model = FastLanguageModel.get_peft_model(
        model,
        r = config['lora']['r'],
        target_modules = config['lora']['target_modules'],
        lora_alpha = config['lora']['lora_alpha'],
        lora_dropout = config['lora']['lora_dropout'],
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    print("Prepare Dataset")
    df = pd.read_csv(config['data']['train_path'])
    unique_intents = df[['label', 'name_intent']].drop_duplicates().sort_values('label')
    class_list_str = "\n".join([f"{row['name_intent']}" for _, row in unique_intents.iterrows()])
    
    raw_dataset = Dataset.from_pandas(df)
    split_dataset = raw_dataset.train_test_split(test_size=0.05)
    
    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        inputs = examples["text"]
        outputs = examples["name_intent"]
        texts = []
        
        for input_text, output_text in zip(inputs, outputs):
            text = PROMPT_TEMPLATE.format(class_list_str, input_text, output_text) + EOS_TOKEN
            texts.append(text)
            
        return { "formatted_text" : texts }

    train_dataset = split_dataset["train"].map(formatting_prompts_func, batched=True)
    eval_dataset = split_dataset["test"].map(formatting_prompts_func, batched=True)

    # Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        learning_rate = float(config['training']['learning_rate']),
        num_train_epochs = config['training']['num_train_epochs'],
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        
        logging_steps = 20,
        eval_strategy = "steps",
        eval_steps = 20,
        save_steps=40,
        
        output_dir = config['training']['output_dir'],
        optim = config['training']['optimizer'],
        seed = 42,
        remove_unused_columns = True,
        report_to="none"
    )
    
    # SFTT Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "formatted_text",
        max_seq_length = config['model']['max_seq_length'],
        args = training_args,
        packing = False,
    )

    print("Training Model")
    trainer.train()

    print("Saving Model")
    model.save_pretrained('/kaggle/working/banking_model/') # config path
    tokenizer.save_pretrained('/kaggle/working/banking_model/')
    
if __name__ == "__main__":
    main()