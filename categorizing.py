#!/usr/bin/env python3
"""
Statement Rephrasing Script for Knowledge/Non-Knowledge Dataset Creation

This script loads a CSV file containing GitHub issue statements, uses an MLX-LM model
to rephrase them into multiple variations, validates the JSON output, and saves
the expanded dataset to a new CSV file.
"""

import logging
import json
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import configparser
import sys
from prompts import PROMPTS
from tqdm import tqdm

try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
except ImportError:
    print("Error: mlx-lm library not found. Please install it with: pip install mlx-lm")
    sys.exit(1)

config = configparser.ConfigParser()
config.read('config.ini')

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rephrasing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_model_and_tokenizer(model_name: str):
    """Load the MLX-LM model and tokenizer."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model: {model_name}")
    weights_directory = config[model_name]['weights_directory']
    try:
        model, tokenizer = load(weights_directory)
        logger.info("Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


def validate_response(response: str, types: List[str]) -> str or None:
    """Validate that the response is valid JSON with the expected structure."""
    logger = logging.getLogger(__name__)

    try:
        if response not in types:
            logger.warning(f"Response \"{response}\" not in Types.")
            return None

        logger.debug("Valid Response.")
        return response
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error validating JSON: {e}")
        return None


def categorize_statement(model, tokenizer, statement: str, types: List[str], max_retries: int = 3) -> str:
    """Rephrase a single statement using the MLX-LM model."""
    logger = logging.getLogger(__name__)

    messages = [
        {
            "role": "system",
            "content": PROMPTS["CATEGORIZE_NON_KNOWLEDGE_SYSTEM_PROMPT"]
        },
        {
            "role": "user",
            "content": PROMPTS["CATEGORIZE_NON_KNOWLEDGE_INPUT_PROMPT"].format(statement=statement)
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    #logger.info(f"PROMPT: {messages}")

    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempt {attempt + 1} for statement rephrasing")

            sampler = make_sampler(temp=0.5, top_p=0.95, top_k=20)
            # Generate response
            response = generate(
                model,
                tokenizer,
                sampler=sampler,
                prompt=prompt,
                max_tokens=4096
            )
            #logger.info(response)
            response_split = response.split("<|end|><|start|>assistant<|channel|>final<|message|>")
            #response_split = response.split("</think>")
            thinking_content = response_split[0]
            content = response_split[1]

            #logger.info(f"Generated response (length: {len(response)})")

            # Validate Response
            #TODO: Improve Case sensitvity
            #validated_data = validate_response(response, types)

            #if validated_data is not None:
            #statements = validated_data
            #logger.info(f"Successfully categorized.")
            return content

            logger.warning(f"Attempt {attempt + 1} failed validation, retrying...")
            #time.sleep(1)  # Brief delay before retry

        except Exception as e:
            logger.error(f"Error in attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                break
                #raise
            #time.sleep(2)

    logger.error(f"Failed to generate valid response after {max_retries} attempts")
    return "FAILED"


def process_csv(input_file: str, output_file: str, model_name: str,
                statement_column: str = "statement", category_column: str = "category"):
    """Process the entire CSV file."""
    logger = logging.getLogger(__name__)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load CSV
    logger.info(f"Loading CSV file: {input_file}")
    input_path = f"datasets/{input_file}.csv"
    logger.info(input_path)
    try:
        df = pd.read_csv(input_path)
        #df = df.sample(frac=0.01)
        df = df[df["ROGUE_precision"] < 0.5]
        logger.info(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        raise

    # Prepare output data
    output_data = []
    total_statements = len(df)

    #Define Non-knowledge Types
    types = ["Unknown unknowns", "Known unknowns", "Knowable known unknowns", "Unknown knowns", "Errors", "Denials", "Neutral", "Knowledge", "ELSE"]

    for i, (idx, row) in tqdm(enumerate(df.iterrows()), total=total_statements):
        #logger.info(f"Processing statement {i + 1}/{total_statements}")

        original_statement = row[statement_column]
        # category_old = row.get(category_column, "unknown")
        # repo = row.get("Repo")
        # type = row.get("GitHub_Type") # issue, PR
        # git_nr = row.get("Nr") # isuue_nr, PR_nr

        try:
            # if category_old != "Knowledge":
            #     category = categorize_statement(model, tokenizer, original_statement, types)
            # else:
            #     category = "Knowledge"
            category = categorize_statement(model, tokenizer, original_statement, types)
            output_row = {}
            # output_row = {
            #     'Repo': repo,
            #     'GitHub_Type' : type,
            #     'Nr' : git_nr,
            #     'original_statement': original_statement,
            #     "Non_Knowledge_old": category_old,
            #     #'rephrased_statement': rephrased,
            #     'Non_knowledge_Type_Roberts': category
            # }
            # Add any additional columns from original row
            for col in df.columns:
                #if col not in [statement_column, category_column]:
                output_row[f'{col}'] = row[col]
            output_row["Label_gpt-120b"] = category

            output_data.append(output_row)

            #logger.info(f"Added category for statement {i+1}")

        except Exception as e:
            logger.error(f"Failed to process row {idx+1}: {e}")
            continue

    # Save output CSV
    if output_data:
        output_path = f"datasets/{output_file}.csv"
        logger.info(f"Saving {len(output_data)} rephrased statements to: {output_path}")
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_path, index=False)
        logger.info("Output CSV saved successfully")

    else:
        logger.error("No valid categories generated")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Categorize GitHub issue statements using MLX-LM")
    parser.add_argument("input_csv", help="Input CSV file with statements")
    parser.add_argument("output_csv", help="Output CSV file for categorized statements")
    parser.add_argument("--model", default="models/Llama-2-7b-hf",
                        help="HuggingFace model name or Path")
    parser.add_argument("--statement-column", default="statement",
                        help="Name of the column containing statements (default: statement)")
    parser.add_argument("--category-column", default="category",
                        help="Name of the column containing categories (default: category)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    # Validate input file exists
    # if not Path(args.input_csv).exists():
    #     logger.error(f"Input file does not exist: {args.input_csv}")
    #     sys.exit(1)

    # Create output directory if needed
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting statement categorising")
    logger.info(f"Input: {args.input_csv}")
    logger.info(f"Output: {args.output_csv}")
    logger.info(f"Model: {args.model}")

    try:
        process_csv(
            input_file=args.input_csv,
            output_file=args.output_csv,
            model_name=args.model,
            statement_column=args.statement_column,
            category_column=args.category_column
        )
        logger.info("Process completed successfully")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()