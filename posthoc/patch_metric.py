import torch
import sys
sys.path.insert(0, "..")
import eval.eval_metrics2 as metrics
from datasets import Dataset
import pandas as pd
import argparse
import warnings

def compute(metric_name, ds, batch_size):
    # Assign metric names to functions
    metric_fns = {
        "bertscore": metrics.get_bertscore_metrics,
        "redundancy": metrics.get_redundancy_scores,
        "alignscore": metrics.eval_alignscore_batch,
        "summac": metrics.eval_summac_batch
    }
    func = metric_fns[metric_name]

    # Redundancy is only measured in aggregate
    if metric_name == "redundancy":
        _, _, _, _ = func(ds["predicted_summary"])
    # BERTScore is batched and always requires preds + refs
    elif metric_name == "bertscore":
        ds =ds.map(
            lambda ex: func(ex["predicted_summary"], ex["summary"])
        )
    else:
    # AlignScore and SummaC use text and predicted_summary
        ds = ds.map(
            func,
            batched=True,
            batch_size=batch_size
        )
    return ds

def load_data(source_file:str, metric_name:str):
    # Read as dataframe
    df = pd.read_csv(source_file)
    if metric_name in df.columns:
        warnings.warn("Metric already exists in source file; this job will overwrite it!")
    # Convert to HF Dataset
    return Dataset.from_pandas(df)

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
    data = load_data(args.file, args.target)

    # Compute metric
    data = compute(args.metric, data, args.batch_size)

    # Overwrite file
    data.to_csv(args.file)
    return

if __name__ == "__main__":
    main()