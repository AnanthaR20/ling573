import torch
import sys
sys.path.insert(0, ".")
import eval.eval_metrics2 as metrics
from datasets import Dataset, load_dataset
import pandas as pd
import argparse
import warnings

def compute(metric_name: str, is_gold: bool, ds: Dataset, batch_size:int):
    """
    Args:
        metric_name: metric to generate
        is_gold: whether the target column is 'summary' or 'predicted_summary'
        ds: Dataset
        batch_size: batch size, if mapping
    Returns:
        Dataset with new metric column
    """
    # Assign metric names to functions
    metric_fns = {
        "bertscore": metrics.get_bertscore_metrics,
        "redundancy": metrics.get_redundancy_scores,
        "alignscore": metrics.eval_alignscore_tmp,
        "summac": metrics.eval_summac_batch
    }
    func = metric_fns[metric_name]

    if is_gold:
        target_column = "summary"
        data_hf = load_dataset("billsum", split="test")
        ds = ds.add_column("text", data_hf["text"])
    else:
        target_column = "predicted_summary"
    
    # Redundancy is only computed in aggregate
    if metric_name == "redundancy":
        _, _, _, _ = func(ds[target_column])
    # BERTScore always requires preds + refs
    elif metric_name == "bertscore":
        ds = ds.map(
            lambda ex: func(ex["predicted_summary"], ex["summary"]),
            batched=True,
            batch_size=batch_size
        )
    else: # Alignscore and SummaC compare texts to target column
        ds = ds.map(
            lambda ex: func(ex["text"], ex[target_column]),
            batched=True,
            batch_size=batch_size
        )
    if is_gold:
        # Remove the mounted text column
        ds = ds.remove_columns(column_names="text")
    return ds

def load_data(source_file:str, metric_name:str):
    """
    Args:
        source_file: source filepath
        metric_name: metric to generate and 'patch'
    Returns:
        Huggingface Dataset for mapping metric and boolean to represent if data is 'gold'
    """
    # Check if this is gold or prediction data
    is_gold = "gold" in source_file
    # Read as dataframe
    df = pd.read_csv(source_file)
    if metric_name in df.columns:
        warnings.warn("Metric already exists in source file; this job will overwrite it!")
    # Convert to HF Dataset
    return Dataset.from_pandas(df), is_gold

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="File to add metrics")
    parser.add_argument("--metric", type=str, help="Specify metric to compute")
    parser.add_argument("--batch_size", type=int, default=8, help="Specify batch size")
    args = parser.parse_args()
    return args

def main():
    args = load_args()

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on ", device)

    # Load input file
    data, is_gold = load_data(args.file, args.metric)

    # Compute metric
    data = compute(args.metric, is_gold, data, args.batch_size)

    # Overwrite file
    data.to_csv(args.file)
    return

if __name__ == "__main__":
    main()