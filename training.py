import logging
import sys
import configparser
import numpy as np
import pandas as pd
from datasets import Dataset 
import torch as t
import torch.nn as nn
from transformers import (
    Trainer, 
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from evaluate import load

# --- Setup ---
config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    pass 

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def run_training():
    # Prepare Data and Labels
    # ------------------------------------------------------------------
    logger.info("Loading CSV files...")
    df_train = pd.read_csv("datasets/knowledge_non_knowledge_v3_training_final_group_balanced.csv")
    df_eval = pd.read_csv("datasets/knowledge_non_knowledge_v3_categorized.csv")

    df_train = df_train.rename(columns={'final_label': 'label'})
    df_eval = df_eval.rename(columns={'final_label': 'label'})
    
    # Ensure 'label' column is of integer type
    df_train['label'] = df_train['label'].astype(int)
    df_eval['label'] = df_eval['label'].astype(int)

    num_labels = df_train['label'].nunique()
    logger.info(f"Detected {num_labels} labels (0, 1, 2).")

    train_dataset = Dataset.from_pandas(df_train)
    eval_dataset = Dataset.from_pandas(df_eval)

    # Load model and tokenizer
    # ------------------------------------------------------------------
    model_name = "roberta-base"
    
    # Fallback if config not found
    weights_directory = config[model_name]['weights_directory'] if model_name in config else model_name
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(weights_directory)
        # Pass label mappings so the model config saves them for inference later
        model = AutoModelForSequenceClassification.from_pretrained(
            weights_directory, 
            num_labels=num_labels,
            torch_dtype=t.bfloat16, 
            device_map="mps"
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Tokenize Data
    # ------------------------------------------------------------------
    def tokenize_function(examples):
        return tokenizer(
            examples["rephrased_statement"], 
            truncation=True, 
            max_length=512,
            padding=False # Important: No padding here for efficiency
        )

    logger.info("Tokenizing datasets...")
    # batched=True uses multi-threading
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # Remove unnecessary columns
    cols_to_keep = ['input_ids', 'attention_mask', 'label']
    tokenized_train = tokenized_train.select_columns(cols_to_keep)
    tokenized_eval = tokenized_eval.select_columns(cols_to_keep)
    
    # Training Config
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./ignorance_classifier_group",           
        eval_strategy="steps",
        eval_steps=10,                  
        save_steps=10,     
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        warmup_steps=10,               
        per_device_train_batch_size=64, 
        per_device_eval_batch_size=64,
        num_train_epochs=2,              
        weight_decay=0.01,
        optim="adamw_torch",            
        load_best_model_at_end=True,     
        run_name="test_roberta_1",
        metric_for_best_model="f1",     
        save_total_limit=2,             # Only keep the 2 best checkpoints to save disk space
        fp16=False,
        bf16=True,                     # MPS doesn't support fp16 AMP well yet, usually uses bf16 or fp32
        use_mps_device=True #if torch.backends.mps.is_available() else False
    )

    metric = load("f1") # load_metric is deprecated, use load from 'evaluate'

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return metric.compute(predictions=predictions, references=labels, average="weighted") # 'weighted' because of multi-class

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 6Train
    # ------------------------------------------------------------------
    logger.info("Starting Training...")
    trainer.train()
    
    logger.info("Saving final model...")
    trainer.save_model("./final_model")
    
    metrics = trainer.evaluate()
    logger.info(f"Final Evaluation Metrics: {metrics}")

if __name__ == "__main__":
    run_training()