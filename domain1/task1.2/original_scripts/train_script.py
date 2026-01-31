# Assignment Part 4

import argparse
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker environment variables
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )
    parser.add_argument(
        "--training-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    data_path = os.path.join(args.training_dir, "financial_qa_dataset.csv")
    df = pd.read_csv(data_path)

    # Format instruction-style training text
    def format_instruction(row):
        return f"Question: {row['question']}\nAnswer: {row['answer']}"

    df["text"] = df.apply(format_instruction, axis=1)
    dataset = Dataset.from_pandas(df[["text"]])

    # ------------------------------------------------------------------
    # Load model & tokenizer
    # ------------------------------------------------------------------
    model_name = "distilgpt2"  # Small model for learning/demo
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT2-style models do not have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        # Labels = input_ids for causal language modeling
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # ------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        fp16=False,  # Set True if using GPU that supports it
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # ------------------------------------------------------------------
    # Train & save
    # ------------------------------------------------------------------
    trainer.train()
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)


if __name__ == "__main__":
    main()



# # Main PEFT Methods 
# 1️. LoRA (Low-Rank Adaptation)
# 2️. Prefix Tuning
# 3️. Prompt Tuning
# 4️. Adapter Layers
# 5️. IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
# 6️. BitFit


# # 1. PEFT library
# from peft import LoraConfig, get_peft_model

# # 2. LoRA configuration
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     target_modules=["c_attn"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# # 3. Wrap the model
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
