#!/usr/bin/env python3
"""Script to evaluate spam classification performance of different models.

This script takes a directory containing label files from different models and a ground truth file,
calculates spam classification metrics, and outputs the results to a CSV file.
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import logging
import glob
from datetime import datetime

# Set up more verbose logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_ground_truth(ground_truth_path: Path) -> pd.DataFrame:
    """Load ground truth spam labels."""
    try:
        logger.info(f"Loading ground truth file from: {ground_truth_path}")
        ground_truth = pd.read_csv(ground_truth_path)
        required_columns = {'id', 'is_spam'}
        if not all(col in ground_truth.columns for col in required_columns):
            raise ValueError(f"Ground truth file must contain columns: {required_columns}")
        logger.info(f"Successfully loaded ground truth with {len(ground_truth)} entries")
        return ground_truth
    except Exception as e:
        logger.error(f"Error loading ground truth file: {e}")
        raise

def load_model_predictions(label_file: Path) -> pd.DataFrame:
    """Load model predictions and convert to spam labels."""
    try:
        logger.info(f"Loading predictions from: {label_file}")
        df = pd.read_csv(label_file)
        logger.info(f"Found {len(df)} predictions")
        # Convert model predictions to binary spam labels (conversation_id = 0 means spam)
        df['is_spam'] = (df['conversation_id'] == 0).astype(int)
        logger.info(f"Identified {df['is_spam'].sum()} spam messages")
        return df[['message_id', 'is_spam']]
    except Exception as e:
        logger.error(f"Error loading predictions from {label_file}: {e}")
        raise

def calculate_metrics(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Calculate spam classification metrics."""
    # Merge ground truth with predictions
    merged = pd.merge(
        ground_truth, 
        predictions,
        left_on='id',
        right_on='message_id',
        how='inner',
        suffixes=('_true', '_pred')
    )
    
    logger.info(f"Merged dataset has {len(merged)} entries")
    
    # Drop any rows with NaN values
    merged = merged.dropna(subset=['is_spam_true', 'is_spam_pred'])
    logger.info(f"After dropping NaN values: {len(merged)} entries")
    
    # Calculate metrics
    y_true = merged['is_spam_true']
    y_pred = merged['is_spam_pred']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    logger.info("Calculated metrics:")
    for metric, value in metrics.items():
        logger.info(f"- {metric}: {value:.4f}")
    
    return metrics

def extract_model_name(label_file: Path) -> str:
    """Extract model name from label file name."""
    # Expected format: labels_YYYYMMDD_modelname_groupname.csv
    parts = label_file.stem.split('_')
    if len(parts) >= 3:
        return parts[2]  # Get the model name part
    return label_file.stem  # Fallback to full stem if pattern doesn't match

def evaluate_spam_detection(group_dir: str) -> None:
    """Evaluate spam detection performance for all models in the directory."""
    logger.info(f"Starting spam detection evaluation for directory: {group_dir}")
    
    group_path = Path(group_dir)
    if not group_path.exists():
        raise ValueError(f"Directory does not exist: {group_dir}")
    
    # Get group name from directory path
    group_name = group_path.name
    logger.info(f"Group name: {group_name}")
    
    # Find ground truth file with group name
    ground_truth_path = group_path / f"GT_spam_{group_name}.csv"
    if not ground_truth_path.exists():
        raise ValueError(f"Ground truth file not found: {ground_truth_path}")
    
    # Load ground truth
    logger.info(f"Loading ground truth from {ground_truth_path}")
    ground_truth = load_ground_truth(ground_truth_path)
    
    # Find all label files
    label_files = list(group_path.glob("labels_*.csv"))
    logger.info(f"Found {len(label_files)} label files")
    if not label_files:
        raise ValueError(f"No label files found in {group_dir}")
    
    # Calculate metrics for each model
    results = []
    for label_file in label_files:
        model_name = extract_model_name(label_file)
        logger.info(f"\nProcessing predictions from model: {model_name}")
        
        try:
            predictions = load_model_predictions(label_file)
            metrics = calculate_metrics(ground_truth, predictions)
            
            # Add model name and file info to metrics
            metrics['model'] = model_name
            metrics['label_file'] = label_file.name
            results.append(metrics)
            
        except Exception as e:
            logger.error(f"Error processing {label_file}: {e}")
            continue
    
    if not results:
        raise ValueError("No results could be calculated from any label file")
    
    # Create results DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    
    # Reorder columns to put model first
    columns = ['model', 'label_file', 'accuracy', 'precision', 'recall', 'f1']
    results_df = results_df[columns]
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Save results with timestamp
    output_file = group_path / f"metrics_spam_detection_{group_name}_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nResults summary:")
    logger.info(f"\n{results_df.to_string()}")
    logger.info(f"\nSaved metrics to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Calculate spam classification metrics for different models.'
    )
    parser.add_argument(
        'group_dir',
        help='Path to directory containing label files and ground truth'
    )
    
    args = parser.parse_args()
    evaluate_spam_detection(args.group_dir)

if __name__ == "__main__":
    main() 