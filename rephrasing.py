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
from random import random
from collections import Counter
import math
import random
import time
import configparser
import sys
from prompts import DISCOURSE_KNOWLEDGE, DISCOURSE_NON_KNOWLEDGE, FORMALITIES, STRUCTURAL, PROMPTS, NEUTRALIZATION_STRATEGIES

try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
except ImportError:
    print("Error: mlx-lm library not found. Please install it with: pip install mlx-lm")
    sys.exit(1)

config = configparser.ConfigParser()
config.read('config_nk.ini')

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
        model, tokenizer = load("/Users/giulianowietig/PycharmProjects/models/Qwen3-Next-80B-A3B-Instruct")
        logger.info("Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


def validate_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Validate that the response is valid JSON with the expected structure."""
    logger = logging.getLogger(__name__)
    target_key = "rephrased_statements" #rephrased_statements
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
    
def calculate_rouge_l(reference: str, candidate: str):
    """
    Calculate ROUGE-L score between reference and candidate strings.
    ROUGE-L measures the longest common subsequence (LCS) between texts.
    
    Args:
        reference: Reference text string
        candidate: Candidate text string
    
    Returns:
        dict with 'precision', 'recall', and 'f1' scores
    """
    def lcs_length(s1, s2):
        """Calculate longest common subsequence length using dynamic programming"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # Tokenize by splitting on whitespace
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    lcs_len = lcs_length(ref_tokens, cand_tokens)
    
    # Calculate precision, recall, and F1
    precision = lcs_len / len(cand_tokens) if len(cand_tokens) > 0 else 0
    recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_bleu(reference: str, candidate: str, max_n: int=10):
    """
    Calculate BLEU score (BP * geometric mean of precisions) between reference and candidate.
    
    Args:
        reference: Reference text string
        candidate: Candidate text string
        max_n: Maximum n-gram size (default: 4)
    
    Returns:
        float: BLEU score (BP * bleu_mean)
    """
    def get_ngrams(tokens, n):
        """Generate n-grams from token list"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    # Tokenize
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    
    if len(cand_tokens) == 0:
        return 0.0
    
    # Calculate brevity penalty (BP)
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0.0
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, min(max_n + 1, cand_len + 1)):
        ref_ngrams = Counter(get_ngrams(ref_tokens, n))
        cand_ngrams = Counter(get_ngrams(cand_tokens, n))
        
        # Count clipped matches
        matches = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        
        if total > 0:
            precisions.append(matches / total)
        else:
            precisions.append(0.0)
    
    # Handle zero precisions (avoid log(0))
    if any(p == 0 for p in precisions):
        return 0.0
    
    # Calculate geometric mean of precisions
    log_precisions = [math.log(p) for p in precisions]
    bleu_mean = math.exp(sum(log_precisions) / len(log_precisions))
    
    # BLEU score = BP * geometric mean
    return bp * bleu_mean
    
def sample_with_weighted_diversity(n_samples: int = 5, knowledge: bool = False):
    samples = []
    # if knowledge:
    #     DISCOURSE = DISCOURSE_KNOWLEDGE
    # else:
    #     DISCOURSE = DISCOURSE_NON_KNOWLEDGE
    # dicts = [FORMALITIES, DISCOURSE, STRUCTURAL]
    dicts = [NEUTRALIZATION_STRATEGIES, FORMALITIES, STRUCTURAL]
    all_values = [list(d.keys()) for d in dicts]
    
    # Track usage counts for each value in each dimension
    usage_counts = [{v: 0 for v in values} for values in all_values]
    
    for _ in range(n_samples):
        combination = []
        for dim_idx, values in enumerate(all_values):
            # Weight by inverse usage (least used items get higher probability)
            counts = usage_counts[dim_idx]
            min_count = min(counts.values())
            # Items with min_count get weight 2, others get weight 1
            weights = [2 if counts[v] == min_count else 1 for v in values]
            
            chosen = random.choices(values, weights=weights, k=1)[0]
            combination.append(chosen)
            usage_counts[dim_idx][chosen] += 1
        
        samples.append(tuple(combination))
    
    return samples

def rephrase_statement(model, tokenizer, system_prompt: str, statement: str, label: str, max_retries: int = 3) -> List[str]:
    """Rephrase a single statement using the MLX-LM model."""
    logger = logging.getLogger(__name__)

    messages = [
        {
            "role": "system",
            "content": system_prompt #PROMPTS["REPHRASE_SYSTEM_PROMPT"]
        },
        {
            "role": "user",
            "content": PROMPTS["REPHRASE_INPUT_PROMPT"].format(statement=statement, category=label)
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    #logger.info(f"PROMPT: {messages}")

    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempt {attempt + 1} for statement rephrasing")
            sampler = make_sampler(temp=0.8, top_p=0.95, top_k=0)
            # Generate response
            response = generate(
                model,
                tokenizer,
                sampler=sampler,
                prompt=prompt,
                max_tokens=8192
            )
            #logger.info(response)
            #response_split = response.split("<|end|><|start|>assistant<|channel|>final<|message|>")
            #response_split = response.split("</think>")
            #thinking_content = response_split[0]
            #content = response_split[1]
            content = response

            logger.debug(f"Generated response (length: {len(response)})")
            #logger.debug(f"Thinking: {thinking_content}")
            logger.debug(f"Final Result: {content}")

            # Validate JSON
            validated_data = validate_json_response(content)

            if validated_data is not None:
                statements = validated_data["rephrased_statements"] #rephrased_example
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


def process_csv(input_file: str, output_file: str, model_name: str, num_styles: int = 5,
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
        FSE_category = row.get("Non_Knowledge_label_FSE_paper_Giuliano")
        # if FSE_category == "Knowledge":
        #     styles = sample_with_weighted_diversity(num_styles, knowledge=True)
        #     DISCOURSE = DISCOURSE_KNOWLEDGE
        # else:
        #     styles = sample_with_weighted_diversity(num_styles, knowledge=False)
        #     DISCOURSE = DISCOURSE_NON_KNOWLEDGE
        styles = sample_with_weighted_diversity(num_styles)
        for j in range(0, num_styles-1):
            logger.info(f"Processing statement {i + 1}/{total_statements}")
            neutralization_strategy = NEUTRALIZATION_STRATEGIES[styles[j][0]]
            #logger.info(styles[j])
            formality = FORMALITIES[styles[j][1]]
            #discourse = DISCOURSE[styles[j][1]]
            structural = STRUCTURAL[styles[j][2]]

            #system_prompt = PROMPTS["REPHRASE_SYSTEM_PROMPT"].format(formality=formality, discourse=discourse, structural=structural)
            system_prompt = PROMPTS["NEUTRALIZATION_SYSTEM_PROMPT"].format(neutralization_strategy=neutralization_strategy, formality=formality, structural=structural)
            #logger.info(f"System Prompt: {system_prompt}")

            original_statement = row[statement_column]
            #logger.info(original_statement)
            category = row.get(category_column) #"unknown"

            # if category == 1:
            #     category = "toxic"
            # else:
            #     category = "healthy"
            repo = row.get("Repo")
            type = row.get("GitHub_Type") # issue, PR
            git_nr = row.get("Nr") # isuue_nr, PR_nr
            FSE_category = row.get("Non_Knowledge_label_FSE_paper_Giuliano")
            long_short = row.get("long_short")

            try:
                rephrased_statements = rephrase_statement(model, tokenizer, system_prompt, original_statement, category)

                if rephrased_statements:
                    # Add each rephrased statement as a new row
                    for rephrased in rephrased_statements:
                        rogue = calculate_rouge_l(reference=original_statement, candidate=rephrased)
                        logger.debug(rogue)
                        #bleu = calculate_bleu(reference=original_statement, candidate=rephrased)
                        #logger.debug(bleu)
                        output_row = {
                            'Repo': repo,
                            'GitHub_Type' : type,
                            'Nr' : git_nr,
                            'original_statement': original_statement,
                            'rephrased_statement': rephrased,
                            'Non_Knowledge_label_FSE_paper_Giuliano': FSE_category, #"Neutral"
                            f'{category_column}': "Neutral",
                            'long_short': long_short,
                            "ROGUE_precision": rogue["precision"],
                            "ROGUE_recall": rogue["recall"],
                            "ROGUE_f1": rogue["f1"],
                            #"BLEU": bleu,
                            "styles": styles[j]
                        }
                        # Add any additional columns from original row
                        for col in df.columns:
                            if col not in [statement_column, category_column]:
                                output_row[f'original_{col}'] = row[col]

                        output_data.append(output_row)

                    logger.info(f"Added {len(rephrased_statements)} rephrased statements for statement {i+1}: style {j+1}")
                else:
                    logger.warning(f"No valid rephrased statements generated for statement {i+1}: style {j+1}")

            except Exception as e:
                logger.error(f"Failed to process row {i+1}: {e}")
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
    parser.add_argument("output_csv", help="Output CSV file for rephrased statements")
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit",
                        help="HuggingFace model name (default: mlx-community/Llama-3.2-3B-Instruct-4bit)")
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

    logger.info("Starting statement rephrasing process")
    logger.info(f"Input: {args.input_csv}")
    logger.info(f"Output: {args.output_csv}")
    logger.info(f"Model: {args.model}")
    
    try:
        process_csv(
            input_file=args.input_csv,
            output_file=args.output_csv,
            model_name=args.model,
            num_styles=10,
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