import logging
import sys
import numpy as np
import pandas as pd
import torch as t
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding
)
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# --- Setup Logging ---
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class ModelEvaluator:
    """Comprehensive evaluation for multi-class text classification"""
    
    def __init__(self, model_path: str, test_csv: str, label_names: list = None):
        """
        Args:
            model_path: Path to saved model directory
            test_csv: Path to test data CSV
            label_names: List of label names (e.g., ['Knowledge', 'Neutral', 'Non-Knowledge'])
        """
        self.model_path = model_path
        self.test_csv = test_csv
        # Labels: 0=Knowledge, 1=Neutral, 2=Non-Knowledge
        self.label_names = label_names or ['Knowledge', 'Neutral', 'Non-Knowledge']
        
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=t.bfloat16,
            device_map="mps"
        )
        self.model.eval()
        
        self.device = t.device("mps" if t.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_and_prepare_data(self):
        """Load and tokenize test data"""
        logger.info(f"Loading test data from {self.test_csv}")
        df_test = pd.read_csv(self.test_csv)
    
        # Rename column if needed
        if 'rephrased_statement' in df_test.columns:
            df_test = df_test.rename(columns={'rephrased_statement': 'statement'})
        
        df_test['final_label'] = df_test['final_label'].astype(int)
        # mapping = {1: 0, 2: 1}
        mapping = {1: 0, 2: 1}
        df_test["final_label"] = df_test["final_label"].replace(mapping)
        
        # Store original data for later analysis
        self.test_df = df_test
        
        # Create dataset
        test_dataset = Dataset.from_pandas(df_test)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["statement"],
                truncation=True,
                max_length=512,
                padding=False
            )
        
        logger.info("Tokenizing test dataset...")
        tokenized_test = test_dataset.map(tokenize_function, batched=True)
        
        cols_to_keep = ['input_ids', 'attention_mask', 'final_label']
        tokenized_test = tokenized_test.select_columns(cols_to_keep)
        
        return tokenized_test
    
    def get_predictions(self, dataset):
        """Get model predictions for the dataset"""
        logger.info("Generating predictions...")
        
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            collate_fn=data_collator
        )
        
        all_preds = []
        all_labels = []
        all_logits = []
        
        with t.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['final_label']
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.cpu()
                
                preds = t.argmax(logits, dim=-1).numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.float().numpy())
                all_logits.extend(logits.float().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_logits)
    
    def compute_overall_metrics(self, y_true, y_pred):
        """Compute overall accuracy, precision, recall, F1"""
        logger.info("Computing overall metrics...")
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1
        }
        
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        logger.info(f"Weighted Precision: {precision:.4f}")
        logger.info(f"Weighted Recall: {recall:.4f}")
        logger.info(f"Weighted F1: {f1:.4f}")
        
        return metrics
    
    def compute_class_specific_metrics(self, y_true, y_pred):
        """Compute per-class precision, recall, F1"""
        logger.info("Computing class-specific metrics...")
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_names,
            output_dict=True
        )
        
        # Print formatted report
        print("\n" + "="*60)
        print("CLASS-SPECIFIC METRICS")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=self.label_names))
        
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot and save confusion matrix"""
        logger.info("Generating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def analyze_misclassifications(self, y_true, y_pred, n_examples=5):
        """Analyze and display misclassified examples"""
        logger.info("Analyzing misclassifications...")
        
        self.test_df['predicted'] = y_pred
        self.test_df['correct'] = (y_true == y_pred)
        
        misclassified = self.test_df[~self.test_df['correct']]
        
        print("\n" + "="*60)
        print(f"MISCLASSIFICATION ANALYSIS")
        print("="*60)
        print(f"Total misclassifications: {len(misclassified)} / {len(self.test_df)}")
        print(f"Error rate: {len(misclassified) / len(self.test_df) * 100:.2f}%\n")
        
        # Show examples for each type of misclassification
        for true_label in range(len(self.label_names)):
            for pred_label in range(len(self.label_names)):
                if true_label == pred_label:
                    continue
                
                subset = misclassified[
                    (misclassified['final_label'] == true_label) & 
                    (misclassified['predicted'] == pred_label)
                ]
                
                if len(subset) > 0:
                    print(f"\n{self.label_names[true_label]} → {self.label_names[pred_label]}: {len(subset)} cases")
                    print("-" * 60)
                    
                    for idx, row in subset.head(15).iterrows():
                        print(f"Text: {row['statement'][:200]}...")
                        print()
        
        return misclassified
    
    def shap_analysis_binary(self, n_samples=100):
        """
        SHAP analysis for binary classification (Knowledge vs Non-Knowledge)
        Ignoring Neutral class for interpretability
        Note: Knowledge=0, Non-Knowledge=2
        """
        logger.info("Starting SHAP analysis (binary: Knowledge vs Non-Knowledge)...")
        
        # Filter dataset to only Knowledge (0) and Non-Knowledge (2)
        binary_df = self.test_df[self.test_df['final_label'].isin([0, 2])].copy()
        binary_df = binary_df.sample(n=min(n_samples, len(binary_df)), random_state=42)
        
        texts = binary_df['statement'].tolist()
        true_labels = binary_df['final_label'].tolist()
        
        # Create prediction function for SHAP
        def predict_proba(texts):
            """Predict probability for binary classification"""
            inputs = self.tokenizer(
                texts,
                truncation=512,
                max_length=512,
                padding=True,
                return_tensors="pt"
            )
            
            with t.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                probs = t.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            # Return only Knowledge (0) and Non-Knowledge (2) probabilities
            return probs[:, [0, 2]]
        
        # Create SHAP explainer
        logger.info("Creating SHAP explainer (this may take a while)...")
        explainer = shap.Explainer(predict_proba, self.tokenizer)
        
        # Compute SHAP values for sample texts
        logger.info(f"Computing SHAP values for {len(texts)} samples...")
        shap_values = explainer(texts[:20])  # Limit to 20 for visualization
        
        # Text plot for first few examples
        print("\n" + "="*60)
        print("SHAP ANALYSIS - BINARY CLASSIFICATION")
        print("="*60)
        
        # Save visualizations
        logger.info("Generating SHAP visualizations...")
        
        # Text plot for individual predictions
        for i in range(min(3, len(texts))):
            plt.figure(figsize=(12, 4))
            shap.plots.text(shap_values[i, :, 1], display=False)  # Class index 1 = Non-Knowledge (label 2)
            plt.savefig(f'shap_text_example_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved SHAP text plot {i}")
        
        # Bar plot showing mean absolute SHAP values
        plt.figure(figsize=(12, 6))
        shap.plots.bar(shap_values[:, :, 1], max_display=20, show=False)  # Non-Knowledge direction
        plt.tight_layout()
        plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved SHAP bar plot")
        
        # Beeswarm plot
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap_values[:, :, 1], max_display=20, show=False)  # Non-Knowledge direction
        plt.tight_layout()
        plt.savefig('shap_beeswarm_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved SHAP beeswarm plot")
        
        logger.info("SHAP analysis complete!")
        
        return shap_values
    
    def run_full_evaluation(self, run_shap=True, shap_samples=100):
        """Run complete evaluation pipeline"""
        logger.info("="*60)
        logger.info("STARTING COMPREHENSIVE MODEL EVALUATION")
        logger.info("="*60)
        
        # Load data
        dataset = self.load_and_prepare_data()
        
        # Get predictions
        y_pred, y_true, logits = self.get_predictions(dataset)
        print(y_true)
        print(y_pred)
        self.test_df['predicted_label'] = y_pred
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        print(probs)
        # self.test_df['prob_knowledge'] = probs[:, 0]
        # self.test_df['prob_neutral'] = probs[:, 1]
        # self.test_df['prob_non_knowledge'] = probs[:, 2]
        # output_csv = self.test_csv.replace('.csv', '_with_predictions.csv')
        # self.test_df.to_csv(output_csv, index=False)
        # logger.info(f"Predictions and logits saved to {output_csv}")

        # Overall metrics
        overall_metrics = self.compute_overall_metrics(y_true, y_pred)
        
        # Class-specific metrics
        class_metrics = self.compute_class_specific_metrics(y_true, y_pred)
        
        # Confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        # Misclassification analysis
        misclassified_df = self.analyze_misclassifications(y_true, y_pred)
        
        # SHAP analysis
        if run_shap:
            shap_values = self.shap_analysis_binary(n_samples=shap_samples)
        
        # Save results
        results = {
            'overall_metrics': overall_metrics,
            'class_specific_metrics': class_metrics
        }
        
        # Save to file
        with open('evaluation_results_roberta_nk_base_v8_openvox.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write("Overall Metrics:\n")
            for metric, value in overall_metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\n" + classification_report(y_true, y_pred, target_names=self.label_names))
        
        logger.info("Evaluation complete! Results saved to evaluation_results.txt")
        
        return results


# --- Main Execution ---
if __name__ == "__main__":
    # Configure paths
    MODEL_PATH = "/Users/giulianowietig/PycharmProjects/models/roberta-non-knowledge-v8-base" #final_model, /Users/giulianowietig/PycharmProjects/models/ModernBERT-non-knowledge-v2
    TEST_CSV = "datasets/OpenVox_non_knowledge_eval.csv"  #datasets/knowledge_non_knowledge_v3_categorized.csv.   datasets/OpenVox_non_knowledge_eval.csv.  datasets/knowledge_non_knowledge_v3_rephrased_rogue_gpt_oss_evaluation_new.csv
    LABEL_NAMES = ['Knowledge', 'Non-Knowledge']
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=MODEL_PATH,
        test_csv=TEST_CSV,
        label_names=LABEL_NAMES
    )
    
    # Run evaluation
    results = evaluator.run_full_evaluation(
        run_shap=False,
        #shap_samples=100  # Number of samples for SHAP analysis
    )