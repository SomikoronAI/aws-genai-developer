# Assignment Part 4

"""
Supervised Fine-Tuning (SFT) Script for Causal Language Models using Hugging Face + LoRA.

This script performs Supervised Fine-Tuning (SFT) on a causal language model (e.g., 
distilgpt2) using the Hugging Face Transformers library and PEFT (LoRA).
It is designed to run both locally and in managed environments such as Amazon SageMaker, 
without requiring code changes.

Training data is expected to be a CSV file containing question - answer pairs. Each row 
is converted into an instruction-style text sequence:

"Question: <question>\nAnswer: <answer>"

The model is trained as a causal language model, where the training labels are identical 
to the input token IDs (labels = input_ids), allowing the model to learn next-token 
prediction over the full sequence.

Low-Rank Adaptation (LoRA) is applied to the attention layers of the base model, so that 
only a small subset of parameters is updated during training, making the process compute- 
and memory-efficient.

Key features:
- Supports causal language model fine-tuning (GPT-style models)
- Uses PEFT LoRA for parameter-efficient training in local execution environment
- Automatically handles tokenization, padding, and truncation
- Saves the fine-tuned model and tokenizer to a configurable output directory

Expected training data format (JSON):
- question: string
- answer: string

Example usage (run locally in Windows PS):
python fm_ft_transformers_train.py `
--training-data-dir .\\data `
--model-dir .\\outputs `
--epochs 50 `
--batch-size 32

This script is intended for experimentation and comparison with managed fine-tuning 
workflows (e.g., Amazon Nova SFT with LoRA).
"""


import os
import json
import pandas as pd
import numpy as np
import argparse
import math
from functools import partial

from datasets import Dataset

import torch
import transformers 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)


# Load evaluation metrics
from evaluate import load
bleu_metric = load("bleu")
rouge_metric = load("rouge")



def tokenize_function(examples, tokenizer, max_length, add_special_tokens=False):
    """
    Tokenizes input-output QA examples for causal language model training.

    Each example should have the format:
    - "Question: <question>\nAnswer: <answer>"
    - Mask prompt tokens -100 in labels. The model is only trained to predict the answer tokens.
    - Input sequences are truncated/padded to `max_length`.
    - add_special_tokens to controls tokenizer-specific special tokens (e.g., <bos>, <eos>).

    Args:
    examples (dict): Batch with key "text".
    tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
    max_length (int): Maximum sequence length.
    add_special_tokens (bool, optional): Whether to add tokenizer special tokens. Defaults to False.

    Returns:
    dict: {"input_ids": [...], "attention_mask": [...], "labels": [...]}
    """

    texts = examples["text"]

    input_ids = []
    labels = []
    attention_masks = []

    for text in texts:
        if "\nAnswer:" not in text:
            continue
        else:
            q, a = text.split("\nAnswer:", 1)
            prompt = q + "\nAnswer:"

        # Tokenize full sequence ONCE
        tokenized = tokenizer(
            prompt + a,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=add_special_tokens,
        )

        ids = tokenized["input_ids"]
        mask = tokenized["attention_mask"]

        # Tokenize prompt separately to get prompt length
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=add_special_tokens,
        )["input_ids"]

        prompt_len = min(len(prompt_ids), max_length)

        label_ids = [-100] * prompt_len + ids[prompt_len:]
        label_ids = label_ids[:max_length]  # safety

        input_ids.append(ids)
        labels.append(label_ids)
        attention_masks.append(mask)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_masks,
    }


def compute_evaluation_metrics_old(trainer):
    """
    Compute perplexity, BLEU, and ROUGE for a causal language model.
    
    Args:
    trainer: Hugging Face Trainer with model, eval_dataset, and tokenizer.
    
    Returns:
    dict: {eval_loss, perplexity, bleu, rouge1, rouge2, rougeL}
    """
    try:
        # Evaluate the model to get logits and loss
        eval_results = trainer.evaluate()
        eval_loss = eval_results.get("eval_loss", None)
        if eval_loss is None:
            print("Evaluation loss not returned by trainer.")
            perplexity = float("inf")
        perplexity = math.exp(eval_loss)
        
        # Generate predictions
        # For causal LM, generate text on eval_dataset
        model = trainer.model
        tokenizer = trainer.tokenizer
        eval_dataset = trainer.eval_dataset
        
        preds = []
        refs = []

        for example in eval_dataset:
            # Reference text
            input_ids = example["input_ids"] 
            # Move tensors to model device
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.unsqueeze(0).to(model.device)

            attention_mask = example.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = torch.tensor(attention_mask)
                attention_mask = attention_mask.unsqueeze(0).to(model.device)
            
            # Generated text
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                num_beams=3,
                max_length=input_ids.shape[1]+64,
                do_sample=False
            )
            
            pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            ref_text  = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            preds.append(pred_text)
            refs.append(ref_text)
        
        # Compute BLEU
        tokenized_preds = [tokenizer.tokenize(p) for p in preds ]
        tokenized_refs  = [[tokenizer.tokenize(r)] for r in refs] 

        bleu_score = bleu_metric.compute(
            predictions=preds,
            references=[[r] for r in refs])["bleu"]

        # Compute ROUGE
        rouge_score = rouge_metric.compute(
            predictions=preds, 
            references=refs)
        
        # Return unified metrics
        return {
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "bleu": bleu_score,
            "rouge1": rouge_score["rouge1"],
            "rouge2": rouge_score["rouge2"],
            "rougeL": rouge_score["rougeL"],
        }
    except OverflowError:
        # Very large loss can overflow
        return {
            "eval_loss": float("inf"),
            "perplexity": float("inf"),
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to compute evaluation metrics: {e}")


def compute_evaluation_metrics(trainer, max_new_tokens=64):
    """
    Compute perplexity, BLEU, and ROUGE for a causal QA model.
    Evaluation is done on ANSWER-ONLY generation.
    """
    try:
        # -----------------------------
        # Perplexity 
        # -----------------------------
        eval_results = trainer.evaluate()
        eval_loss = eval_results.get("eval_loss")
        perplexity = math.exp(eval_loss) if eval_loss is not None else float("inf")

        model = trainer.model
        tokenizer = trainer.tokenizer
        eval_dataset = trainer.eval_dataset

        model.eval()

        preds = []
        refs = []

        for example in eval_dataset:
            # ----------------------------------
            # 1. Recover QUESTION-ONLY prompt
            # ----------------------------------
            input_ids = example["input_ids"]
            labels = example["labels"]

            # Prompt = tokens where label == -100
            prompt_ids = [
                tok_id for tok_id, lab in zip(input_ids, labels) if lab == -100
            ]

            prompt_text = tokenizer.decode(
                prompt_ids, skip_special_tokens=True
            )

            # ----------------------------------
            # 2. Gold answer from labels
            # ----------------------------------
            answer_ids = [
                tok_id for tok_id, lab in zip(input_ids, labels) if lab != -100
            ]

            ref_text = tokenizer.decode(
                answer_ids, skip_special_tokens=True
            ).strip()

            # ----------------------------------
            # 3. Generate answer
            # ----------------------------------
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=3,
                )

            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            pred_text = tokenizer.decode(
                gen_ids, skip_special_tokens=True
            ).strip()

            preds.append(pred_text)
            refs.append(ref_text)

        # -----------------------------
        # BLEU (expects STRINGS)
        # -----------------------------
        bleu = bleu_metric.compute(
            predictions=preds,
            references=[[r] for r in refs]
        )["bleu"]

        # -----------------------------
        # ROUGE
        # -----------------------------
        rouge = rouge_metric.compute(
            predictions=preds,
            references=refs
        )

        return {
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "bleu": bleu,
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
        }
    except OverflowError:
        # Very large loss can overflow
        return {
            "eval_loss": float("inf"),
            "perplexity": float("inf"),
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to compute evaluation metrics: {e}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="./outputs")
    parser.add_argument("--training-data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    return parser.parse_args()


def check_transformers_version():
    transformers_version = transformers.__version__
    torch_version = torch.__version__
    tested_versions =["2.10.0", "4.56.2"]
    if torch_version != tested_versions[0] and transformers_version != tested_versions[1]:
         print("")
         print(f"This script is tested using torch version: {tested_versions[0]}") 
         print(f"This script is tested using transformers version: {tested_versions[1]}") 
    else:
        print("")
        print(f"Matching Torch and Transformers versions found.")
        print("Proceed to the next tasks ... \n")


def main():
    check_transformers_version()
    args = parse_args()

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    try:
        data_files= ["data_fm_finance_qa_01.json", 
                     "data_fm_finance_qa_02.json", 
                     "data_fm_finance_qa_03.json"
                     ]
        
        all_samples = []
        for data_file in data_files:
            data_path = os.path.join(args.training_data_dir, data_file)
            with open(data_path, "rb") as f:
                data = json.load(f)
            all_samples.extend(data)

        np.random.shuffle( all_samples )
        revised_samples  = [{"question":x["question"], "answer":x["ground_truth"]} for x in all_samples]
        df = pd.DataFrame(revised_samples)

    except Exception as e:
        raise RuntimeError(f"Failed to load training data: {e}")

    # --------------------------------------------------
    # Format dataset
    # --------------------------------------------------
    def format_instruction(row):
        return f"Question: {row['question']}\nAnswer: {row['answer']}"

    df["text"] = df.apply(format_instruction, axis=1)
    dataset = Dataset.from_pandas(df[["text"]])

    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # --------------------------------------------------
    # Load tokenizer & model
    # --------------------------------------------------
    # model_name = "distilbert/distilgpt2"
    model_name = "gpt2"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load model/tokenizer: {e}")

    # Gradient checkpointing
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    # --------------------------------------------------
    # Apply LoRA (PEFT)
    # --------------------------------------------------
    # ["c_attn"] : linear layer for GPT-2 family where learning is allowed
    # ["q_proj", "v_proj"] : linear layer for LLaMA/Mistral family where learning is allowed
    # ["q", "v"] : linear layer for T5 family where learning is allowed
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn"], 
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    print("Printing trainable parameters ...")
    model.print_trainable_parameters()


    # --------------------------------------------------
    # Tokenization
    # --------------------------------------------------
    token_func = partial( 
        tokenize_function, 
        tokenizer=tokenizer, 
        max_length=args.max_length,
        add_special_tokens=False
        )
    
    train_dataset = train_dataset.map(
        token_func,
        batched=True,
        remove_columns=["text"],
        )
    eval_dataset = eval_dataset.map(
        token_func,
        batched=True,
        remove_columns=["text"],
        )
    print("Train columns:", train_dataset.column_names)
    print("Eval columns:", eval_dataset.column_names)

    # --------------------------------------------------
    # Training arguments
    # --------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ## For larger dataset
        # logging_strategy="steps", 
        # logging_steps=10,
        # eval_strategy="steps", 
        # # eval_steps=50, 
        # save_strategy="steps", 
        # save_steps=100,
        # save_total_limit=2,
        # report_to="none",
        ## For smaller dataset
        logging_strategy="epoch", 
        eval_strategy="epoch",  
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",

        prediction_loss_only=False, 
        fp16=False,
        dataloader_pin_memory=False, # False if use CPU, True for GPU    
        )

    # --------------------------------------------------
    # Data Collector
    # --------------------------------------------------
    # "pad_to_multiple_of": Controls the padding length of the sequences.
    # So that all sequences in a batch are a multiple of N.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, 
        pad_to_multiple_of=None,   # None if use CPU, 8 or 16 for GPU 
        )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    # --------------------------------------------------
    # Train & save
    # --------------------------------------------------
    trainer.train()
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------
    metrics = compute_evaluation_metrics(trainer)
    print(f"Eval Loss: {metrics['eval_loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {metrics['rougeL']:.4f}")





if __name__ == "__main__":
    main()