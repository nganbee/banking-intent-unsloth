# FINETUNING BANKING INTENT MODEL WITH UNSLOTH

## Overview
Applying fine-tuning techniques to a banking intent classification task using the **BANKING77** dataset and the **Unsloth** library. This repository contains the full pipeline for data preprocessing, model training, and standalone inference.

## Setup environment
Setup the environment on **Kaggle** or **Collab**.
- **Install dependence**
    ```bash
    # Use for running on Kaggle, simple implement
    bash setup.sh

    # Another method
    pip install -r requirements.txt
    ```

## Data Preparation and Processing
- Source: Using [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) dataset.
    - Note: Because the **BANKING77** on **Hugging Face** can not download through the `load_dataset` function. Need to download on these link:  

        > TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"  
        TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

- Sampling: A subset is sampled 50% to computational efficiency.
- Preprocessing: Includes text normalization, label mapping, and splitting into train/test sets. Split the val set in the training phrase.
- Run Preprocessing:
    ```bash
    python scripts/preprocess_data.py
    ```

## Fine-tuning with Unsloth
The model is fine-tuned to classify banking queries accurately into predefined intent labels.

- Parameters in `train.yml`:
    | **Parameter** | **Value** |
    | :----------| :----------:|
    | **Base Model** | `unsloth/Qwen2.5-3B-bnb-4bit` |
    | **Max Sequence Length** | `1024` |
    | **Per Device Train Batch Size** | `4` |
    | **Gradient Accumulation Steps** | `4` |
    | **Learning Rate** | `2e-4` |
    | **Number of Epochs** | `3` |
    | **Optimizer** | `adamw_8bit` |
    | **Weight Decay** | `0.01` |
    | **LR Scheduler Type** | `linear` |
    | **LoRA Rank ($r$)** | `16` |
    | **LoRA Alpha ($\alpha$)** | `16` |
    | **LoRA Dropout** | `0` |
    | **LoRA Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

- Run Training:
    ```bash
    bash train.sh
    ```

- Model Checkpoint: Have been saved into Hugging Face (`imbee510/bank-intent-qwen-unsloth`)

## Inference Implementation
The inference script supports multiple execution modes. You can specify the model type using the `--mode` flag and use the `--interactive` flag for real-time testing.
- Available Modes: `base_zero_shot`, `base_few_shot`, `finetuned`

- Command Examples:
    ```bash
    # Run fine-tuned model not in interactive mode
    bash inference.sh --mode finetuned

    # Run base zero-shot in interactive mode -> use gradio
    bash inference.sh --mode base_zero_shot --interactive
    ```
## Evaluation
To compare the performance of the fine-tuned model against the base model (Zero-shot and Few-shot), use the `evaluate.py` script. This script produces a detailed classification report and saves results to a CSV file
- Evaluate All Models: Use the `all` mode to run evaluations for all three configurations sequentially.
    ```bash
    # Run comprehensive evaluation for all modes
    python scripts/evaluate.py --mode all
    ```

- Evaluate a Specific Mode:
    ```bash
    # Evaluate only the fine-tuned model
    python scripts/evaluate.py --mode finetuned
    ```

- **Outputs**: The script will generate `evaluation_results_[mode].csv` files containing predictions for each test sample.
    
## Video Demonstration
Link of video: [HERE](https://drive.google.com/file/d/1V_tKkDfhWtEVdf4rMiGycCup2rjXPXi6/view?usp=sharing)