import torch
import sys
sys.path.insert(0, ".")
import eval.eval_metrics2 as metrics
from datasets import Dataset
import pandas as pd
import argparse
import warnings

def compute(metric_name: str, 
            is_gold: bool, 
            is_pilot: bool,
            ds: Dataset,
            batch_size: int
        ):
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
        "summac": metrics.eval_summac_batch,
        "lftk": metrics.eval_lftk,
        "rouge": metrics.eval_rouge_patch
    }
    func = metric_fns[metric_name]

    # NLLP predictions have the column predicted_summary
    if not is_gold and not is_pilot:
        target_column = "predicted_summary"
    else:
        # If using the stored gold or baseline data
        # TEMPORARY: hard-coded filepath
        if metric_name == "summac" or metric_name == "alignscore":
            source_data = pd.read_csv("preprocess/nllp_data/billsum_clean_test.csv")
            ds = ds.add_column("text", source_data["text"].tolist())
        if is_gold:
            target_column = "summary"
        if is_pilot:
            target_column = "summary_generated"
            # If reference summary is not in existing columns, add it
            if "summary" not in ds.column_names:
                print("Adding reference summary column...")
                ds = ds.add_column("summary", source_data["summary"].tolist())
    
    print(f"Targeting column {target_column}")
    # Redundancy is only computed in aggregate
    if metric_name == "redundancy":
        _, _, _, _ = func(ds[target_column])
    # BERTScore always requires preds + refs
    elif metric_name == "bertscore" or metric_name == "rouge":
        ds = ds.map(
            lambda ex: func(ex[target_column], ex["summary"]),
            batched=True,
            batch_size=batch_size
        )
    elif metric_name == "lftk":
        ds = ds.map(
        lambda ex: func(ex[target_column], suffix=".GEN"),
        batched=False
    )
    else: 
        torch.cuda.empty_cache()  # Clear any leftover memory usage just in case
        ds = ds.map(
            lambda ex: func(ex['text'], ex[target_column]),
            batched=True,
            batch_size=batch_size
        )
    if (is_gold or is_pilot) and "text" in ds.column_names:
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
    # Check if this is gold, spring deliverable or prediction data
    is_gold = "gold" in source_file
    is_pilot = "deliverable" in source_file or "baseline" in source_file
    if is_gold:
        print(f"Preparing to patch {metric_name} onto gold data")
    if is_pilot:
        print("Preparing to patch {metric_name} onto baseline/pilot data")
    else:
        print(f"Preparing to patch {metric_name} onto NLLP prediction data")

    # Read as dataframe
    df = pd.read_csv(source_file)
    if metric_name in df.columns:
        warnings.warn("Metric already exists in source file; this job will overwrite it!")
    # Convert to HF Dataset
    return Dataset.from_pandas(df), is_gold, is_pilot

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="File to add metrics")
    parser.add_argument("--metric", type=str, help="Specify metric to compute")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    args = load_args()

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on ", device)

    # Load input file
    data, is_gold, is_pilot = load_data(args.file, args.metric)

    # Compute metric
    data = compute(args.metric, is_gold, is_pilot, data, args.batch_size)

    # Overwrite file
    data.to_csv(args.file)
    print("COMPLETE")
    return

if __name__ == "__main__":
    main()