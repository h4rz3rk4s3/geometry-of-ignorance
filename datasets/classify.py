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


def validate_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Validate that the response is valid JSON with the expected structure."""
    logger = logging.getLogger(__name__)
    target_key = "rephrased_example" #rephrased_statements
    try:
        # Try to find JSON in the response (in case there's extra text)
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            logger.warning("No JSON object found in response")
            return None

        json_str = response[start_idx:end_idx]
        data = json.loads(json_str)

        # Validate structure
        if not isinstance(data, dict):
            logger.warning("Response is not a JSON object")
            return None

        if target_key not in data:
            logger.warning("Missing 'rephrased_statements' key in response")
            return None

        if not isinstance(data[target_key], list):
            logger.warning("'rephrased_statements' is not a list")
            return None

        if len(data[target_key]) == 0:
            logger.warning("Empty rephrased_statements list")
            return None

        # Check that all statements are strings
        for i, stmt in enumerate(data[target_key]):
            if not isinstance(stmt, str) or len(stmt.strip()) == 0:
                logger.warning(f"Invalid statement at index {i}")
                return None

        logger.debug(f"Valid JSON with {len(data[target_key])} statements")
        return data

    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error validating JSON: {e}")
        return None


def rephrase_statement(model, tokenizer, statement: str, label: str, max_retries: int = 3) -> List[str]:
    """Rephrase a single statement using the MLX-LM model."""
    logger = logging.getLogger(__name__)

    messages = [
        {
            "role": "system",
            "content": PROMPTS["NEGATE_SYSTEM_PROMPT"]
        },
        {
            "role": "user",
            "content": PROMPTS["NEGATE_INPUT_PROMPT"].format(statement=statement, label=label)
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    #logger.info(f"PROMPT: {messages}")

    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempt {attempt + 1} for statement rephrasing")
            sampler = make_sampler(temp=0.8, top_p=0.95, top_k=20, min_p=0)
            # Generate response
            response = generate(
                model,
                tokenizer,
                sampler=sampler,
                prompt=prompt,
                max_tokens=8192
            )

            response_split = response.split("</think>")
            thinking_content = response_split[0]
            content = response_split[1]

            logger.debug(f"Generated response (length: {len(response)})")
            logger.debug(f"Thinking: {thinking_content}")
            logger.debug(f"Final Result: {content}")

            # Validate JSON
            validated_data = validate_json_response(content)

            if validated_data is not None:
                statements = validated_data["rephrased_example"]
                logger.info(f"Successfully generated {len(statements)} rephrased statements")
                return statements

            logger.warning(f"Attempt {attempt + 1} failed validation, retrying...")
            #time.sleep(1)  # Brief delay before retry

        except Exception as e:
            logger.error(f"Error in attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                break
                #raise
            #time.sleep(2)

    logger.error(f"Failed to generate valid response after {max_retries} attempts")
    return []

def get_classification(model, tokenizer, statement: str):
    batch = tokenizer.encode("You are amazing!", return_tensors="pt")

    output = model(batch)


def process_csv(input_file: str, output_file: str, model_name: str,
                statement_column: str = "statement", category_column: str = "category"):
    """Process the entire CSV file."""
    logger = logging.getLogger(__name__)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load CSV
    logger.info(f"Loading CSV file: {input_file}")
    input_path = f"datasets/{input_file}.csv"
    try:
        df = pd.read_csv(input_path)
        #df = df.sample(frac=0.1).copy()
        logger.info(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        raise

    # Validate required columns
    if statement_column not in df.columns:
        raise ValueError(f"Column '{statement_column}' not found in CSV")

    # Prepare output data
    output_data = []
    total_statements = len(df)

    for i, (idx, row) in enumerate(df.iterrows()):
        logger.info(f"Processing statement {i + 1}/{total_statements}")

        original_statement = row[statement_column]

        try:
            rephrased_statements = rephrase_statement(model, tokenizer, original_statement, category)

            if rephrased_statements:
                # Add each rephrased statement as a new row
                for rephrased in rephrased_statements:
                    output_row = {
                        #'Repo': repo,
                        #'GitHub_Type' : type,
                        #'Nr' : git_nr,
                        'original_statement': original_statement,
                        'rephrased_statement': rephrased,
                        f'{category_column}': category
                    }
                    # Add any additional columns from original row
                    for col in df.columns:
                        if col not in [statement_column, category_column]:
                            output_row[f'original_{col}'] = row[col]

                    output_data.append(output_row)

                logger.info(f"Added {len(rephrased_statements)} rephrased statements for row {i}")
            else:
                logger.warning(f"No valid rephrased statements generated for row {i}")

        except Exception as e:
            logger.error(f"Failed to process row {i}: {e}")
            continue

    # Save output CSV
    if output_data:
        output_path = f"rephrased/{output_file}.csv"
        logger.info(f"Saving {len(output_data)} rephrased statements to: {output_path}")
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_path, index=False)
        logger.info("Output CSV saved successfully")

        # Log summary statistics
        original_count = len(df)
        rephrased_count = len(output_data)
        avg_rephrases = rephrased_count / original_count if original_count > 0 else 0
        logger.info(
            f"Summary: {original_count} original → {rephrased_count} total ({avg_rephrases:.1f} avg per original)")
    else:
        logger.error("No valid rephrased statements generated")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Rephrase GitHub issue statements using MLX-LM")
    parser.add_argument("input_csv", help="Input CSV file with statements")
    #parser.add_argument("output_csv", help="Output CSV file for rephrased statements")
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit",
                        help="HuggingFace model name (default: mlx-community/Llama-3.2-3B-Instruct-4bit)")
    parser.add_argument("--statement-column", default="statement",
                        help="Name of the column containing statements (default: statement)")
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
    output_path = Path(args.input_csv)
    #output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting statement rephrasing process")
    logger.info(f"Input: {args.input_csv}")
    logger.info(f"Output: {args.output_csv}")
    logger.info(f"Model: {args.model}")

    try:
        process_csv(
            input_file=args.input_csv,
            output_file=args.output_csv,
            model_name=args.model,
            statement_column=args.statement_column,
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