"""Evaluate non-knowledge classifiers on labeled CSV files.

  encoder  fine-tuned encoder LMs (RoBERTa, ModernBERT), as reported in the paper
  llm      the instruction-tuned LLMs we used to generate the synthetic data
           (Qwen3-Next-80B-A3B-Instruct, gpt-oss-120b), used zero-shot as
           classifiers.

Labels: 0 = Knowledge, 1 = Non-Knowledge.
Input CSVs need a `final_label` column and a `statement` (or
`rephrased_statement`) column. The SHAP analysis lives in shap_eval.ipynb.

Examples:
  python run_eval_with_llms.py encoder --model /Users/giulianowietig/PycharmProjects/models/roberta-non-knowledge-v8-large
  python run_eval_with_llms.py llm --model Qwen/Qwen3-Next-80B-A3B-Instruct
  python run_eval_with_llms.py llm --model openai/gpt-oss-120b --reasoning-effort medium
"""
#/path/to/roberta-non-knowledge-v8-large
import argparse
import logging
import re
import sys
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

os.environ.setdefault("HF_HOME", "/Volumes/Samsung SSD 990 PRO 4TB/huggingface")

LABEL_NAMES = ["Knowledge", "Non-Knowledge"]

DEFAULT_DATASETS = [
    "datasets/knowledge_non_knowledge_v3_categorized.csv",  # ground truth
    "datasets/OpenVox_non_knowledge_eval.csv",  # OoD
    "datasets/independent_evaluation_set.csv",  # validation
]

SYSTEM_PROMPT = """
    You classify whether single messages from software developer communication (issues, pull requests, chats, emails, Q&A posts) express Knowledge or Non-Knowledge.
    
    # What is (Non-)Knowledge?
     - **Knowledge statements**: Express understanding, provide solutions, share expertise, or make confident assertions about software/technical topics
     - **Non-knowledge statements**: Communicate lack of knowledge through direct questions, confusion about unexpected behavior, uncertainty about whether something is a bug, or requests for clarification
    
    ## Types of Non-Knowledge (ONLY FOR BETTER UNDERSTANDING, DON'T RETURN THEM AS LABELS)
    - **Unknown unknowns**: Represents a state of ignorance, where one is not aware of what is not known (i.e., where one doesn't know that he/she doesn't know). It refers to a total 
    absence of knowledge, such that the state of becoming aware is beyond anticipation. The revelation of such ignorance can be a source of surprise or confusion. This type of 
    non-knowledge is alwys based on an absence of knowledge.
    - **Known unknowns**: Represents a recognized incompleteness of knowledge, where one has an awareness of their non-knowledge (i.e., there are certain things that we know that we 
    do not know). This type of non-knowledge is often revealed through questions (either direct or indirect) or the expression of uncertainty (e.g., whether something is a bug or 
    something is the right way to proceed). This type of non-knowledge is alwys based on an absence of knowledge.
    - **Knowable known unknowns**: Represents a type of non-knowledge which one chooses not to overcome. It refers to Knowledge that is not central or important for one, and is 
    therefore also called rational ignorance. It differs from a known unknown in that it is knowable given sufficient motivation and resources and refers to ignorance that one 
    is not motivated to overcome through the expenditure of the necessary resources to acquire it. This type of non-knowledge is alwys based on an ignorance about existing knowledge.
    - **Unknown knowns**: Unknown knowns denotes knowledge that is unrecognized, such as knowledge embedded in tacit routines, intuition, and organizational practices that neither 
    workforce nor management explicitly acknowledge. It refers to things that we do not know (this is why this is a non-knowledge type) that we know. This type of non-knowledge 
    is alwys based on an ignorance about existing knowledge.
    
    Answer with exactly one label: Knowledge or Non-Knowledge.
  """
USER_PROMPT = 'Message:\n"""\n{text}\n"""\n\nLabel:'


# --- Logging -----------------------------------------------------------------
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# --- Data ----------------------------------------------------------------------
def to_binary(label, is_training: bool = False) -> int:
    label = str(label).strip()
    # OoD uses 0 = K/N, 2 = NK (no 1); GT and v8 use 0 = K/N, 1 = NK
    mapping = {"0": 0, "1": 1} if is_training else {"0": 0, "1":0, "2": 1}
    if label not in mapping:
        raise ValueError(f"Unknown label {label!r} (is_training={is_training})")
    return mapping[label]

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "rephrased_statement" in df.columns:
        df = df.rename(columns={"rephrased_statement": "statement"})
    is_training = any(k in csv_path for k in ("training", "independent", "evaluation"))
    df["final_label"] = df["final_label"].apply(to_binary, is_training=is_training)
    df["final_label"] = df["final_label"].astype(int)
    unexpected = set(df["final_label"].unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"{csv_path}: labels must be 0/1, found {sorted(unexpected)}")
    return df


# --- Backend 1: fine-tuned encoder LMs -----------------------------------------
class EncoderClassifier:
    """Fine-tuned sequence classifier (unchanged behavior from the paper runs)."""

    def __init__(self, model_path: str, batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Loading encoder from {model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()

    def predict(self, texts: list) -> pd.DataFrame:
        probs = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[i : i + self.batch_size],
                    truncation=True,
                    max_length=512,
                    padding=True,  # pad to the longest example in the batch
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**batch).logits.float()
                probs.append(self.torch.softmax(logits, dim=-1).cpu().numpy())
        probs = np.concatenate(probs)
        return pd.DataFrame(
            {
                "predicted_label": probs.argmax(axis=1),
                "prob_knowledge": probs[:, 0],
                "prob_non_knowledge": probs[:, 1],
            }
        )

 
# --- Backend 2: zero-shot LLMs with mlx-lm -------------------------------------
def extract_final_answer(output: str) -> str:
    """Return the answer part of a generation.
 
    gpt-oss (harmony format) writes its reasoning to the "analysis" channel and
    the answer to the "final" channel; Qwen3 models may wrap reasoning in
    <think> tags. Other output is returned unchanged.
    """
    marker = "<|channel|>final<|message|>"
    if marker in output:
        output = output.split(marker, 1)[1]
    elif "<|channel|>analysis" in output:
        return ""  # reasoning was cut off before the final answer
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.S)
    return re.sub(r"<\|[^|]*\|>", "", output).strip()  # drop remaining special tokens
 
 
def parse_label(answer):
    """Map an LLM answer to 0/1. Returns None if no label is found.
 
    Uses the first label mentioned, since the prompt asks for the label only.
    "non-knowledge" is matched before "knowledge" because it contains it.
    """
    if not answer:
        return None
    match = re.search(r"non[\s\-_]*knowledge|knowledge", answer.lower())
    if match is None:
        return None
    return 1 if match.group(0).startswith("non") else 0
 
 
class LLMClassifier:
    """Zero-shot classification with mlx-lm, set up as in rephrasing.py
    (same loader, chat template call, and generate call), but with greedy
    decoding instead of temperature sampling."""
 
    def __init__(self, model_path: str, max_tokens: int = 2048, reasoning_effort: str = None):
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
 
        self.generate = generate
        # temp=0 means argmax decoding: deterministic and reproducible.
        self.sampler = make_sampler(temp=0.0)
        # Reasoning models (gpt-oss) generate their reasoning before the
        # answer, so keep this generous; the answer itself is one word.
        self.max_tokens = max_tokens
        # Data generation used the chat template's default (no argument), so
        # leave this unset to match it.
        self.template_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
 
        logger.info(f"Loading LLM from {model_path}")
        self.model, self.tokenizer = load(model_path)
 
    def _classify(self, text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(text=text)},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, **self.template_kwargs)
        return self.generate(
            self.model, self.tokenizer, prompt=prompt, sampler=self.sampler, max_tokens=self.max_tokens
        )
 
    def predict(self, texts: list) -> pd.DataFrame:
        outputs = []
        for i, text in enumerate(texts, 1):
            outputs.append(self._classify(text))
            if i % 10 == 0 or i == len(texts):
                logger.info(f"Generated {i}/{len(texts)}")
 
        answers = [extract_final_answer(o) for o in outputs]
        parsed = [parse_label(a) for a in answers]
        n_invalid = sum(p is None for p in parsed)
        if n_invalid:
            logger.warning(
                f"{n_invalid}/{len(texts)} answers contained no label; counted as Knowledge "
                "(no detection). See column llm_raw_output; if reasoning was cut off, "
                "increase --max-tokens."
            )
        return pd.DataFrame(
            {
                "predicted_label": [0 if p is None else p for p in parsed],
                "llm_parse_ok": [p is not None for p in parsed],
                "llm_answer": answers,
                "llm_raw_output": outputs,
            }
        )
 
 
# --- Reporting -----------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
 
 
def print_misclassifications(df, n_examples=15):
    wrong = df[df["final_label"] != df["predicted_label"]]
    print(f"\nMisclassifications: {len(wrong)} / {len(df)} ({len(wrong) / len(df) * 100:.2f}%)")
    for true_label, pred_label in [(0, 1), (1, 0)]:
        subset = wrong[(wrong["final_label"] == true_label) & (wrong["predicted_label"] == pred_label)]
        if len(subset):
            print(f"\n{LABEL_NAMES[true_label]} -> {LABEL_NAMES[pred_label]}: {len(subset)} cases")
            print("-" * 60)
            for text in subset["statement"].head(n_examples):
                print(f"Text: {str(text)[:200]}...\n")
 
 
def evaluate(classifier, csv_path: str, model_tag: str, out_dir: Path) -> dict:
    logger.info("=" * 60)
    logger.info(f"Evaluating {model_tag} on {csv_path}")
    df = load_dataset(csv_path)
    predictions = classifier.predict(df["statement"].astype(str).tolist())
    df = pd.concat([df.reset_index(drop=True), predictions], axis=1)
 
    y_true, y_pred = df["final_label"].to_numpy(), df["predicted_label"].to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {
        "n": len(df),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_non_knowledge": f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0),
    }
    if "llm_parse_ok" in df:
        metrics["n_unparsed"] = int((~df["llm_parse_ok"]).sum())
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=LABEL_NAMES, zero_division=0)
 
    prefix = out_dir / f"{model_tag}__{Path(csv_path).stem}"
    df.to_csv(f"{prefix}_predictions.csv", index=False)
    plot_confusion_matrix(y_true, y_pred, f"{prefix}_confusion_matrix.png")
    with open(f"{prefix}_metrics.txt", "w") as f:
        f.write(f"model: {model_tag}\ndataset: {csv_path}\n\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n" if isinstance(value, float) else f"{name}: {value}\n")
        f.write("\n" + report)
 
    print("\n" + report)
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
    print_misclassifications(df)
    logger.info(f"Results written to {prefix}_*")
    return metrics


# --- CLI -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("backend", choices=["encoder", "llm"])
    parser.add_argument("--model", required=True,
                        help="encoder: fine-tuned model directory; llm: MLX weights directory (as in config_nk.ini)")
    parser.add_argument("--data", nargs="+", default=DEFAULT_DATASETS, help="one or more evaluation CSVs")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=64, help="encoder only")
    parser.add_argument("--max-tokens", type=int, default=2048, help="llm only")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                        help="llm only, gpt-oss: reasoning level (unset = template default, as in data generation)")
    args = parser.parse_args()
 
    if args.backend == "encoder":
        classifier = EncoderClassifier(args.model, batch_size=args.batch_size)
    else:
        classifier = LLMClassifier(args.model, max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort)
 
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = re.sub(r"[^\w.-]+", "_", Path(args.model).name)
 
    summary = {csv: evaluate(classifier, csv, model_tag, out_dir) for csv in args.data}
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(out_dir / f"{model_tag}__summary.csv")
    print("\n" + summary_df.round(3).to_string())
 
 
if __name__ == "__main__":
    main()