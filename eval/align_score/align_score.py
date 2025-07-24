"""
This script is for generating alignscore factuality metrics.

Last Updated: 7-18-2025
"""
import argparse
import pandas as pd
from alignscore import AlignScore
from datasets import load_dataset

# print(docs.shape)
# print(type(docs))

# Initialize Model for scoring factuality
scorer = AlignScore(
    model='roberta-base', 
    batch_size=32, 
    device="cpu",#'cuda:0', 
    ckpt_path="https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt",#'AlignScore-base.ckpt', 
    evaluation_mode='nli_sp'
)

def eval_alignscore(contexts:list[str], claims: list[str]) -> dict:
    """Gets a dictionary of AlignScore factuality scores for a 
    claim given some context. Paper at
    https://arxiv.org/abs/2305.16739

    Arguments:
      contexts: list of body of texts the claims are evaluated against. typically will be
      from the bill or gold summary.
      claims: list of generated summaries whose text will be scored against the claim.

    Returns:
      A number in range [0,1] indicating the degree to which the claim
      is factually consistent with the context.
    """
    return scorer.score(contexts, claims)


def main():
    # determine cutoff for subset
    bill_examples_cutoff = None if args.num_examples == None else int(args.num_examples) + 1
    summary_examples_cutoff = None if args.num_examples == None else int(args.num_examples)
    # Assemble the files into corresponding lists
    bills = []
    summaries = []

    if args.summary_file == "gold":
        df = load_dataset("FiscalNote/billsum")
        summaries = list(df['test'].to_pandas()['summary'])[0:summary_examples_cutoff]
    else:
        summaries = list(pd.read_csv(args.summary_file)["predicted_summary"])[0:summary_examples_cutoff]

    with open(args.bill_file,'r') as f:
        bills = f.read().splitlines()[1:bill_examples_cutoff]

    total_bills = len(bills)
    # Iteratively record AlignScore values in case it quits randomly at a certain point.
    with open(args.output_file,'w') as op:
        for i,context,claim in zip(range(len(bills)),bills,summaries):
            align_score_values = eval_alignscore([context],[claim])
            op.write(f"<ROW>{i}</ROW><BILL>{context}</BILL><SUMMARY>{claim}</SUMMARY><BILL_FROM>{args.bill_file}</BILL_FROM><SUMMARY_FROM>{args.summary_file}</SUMMARY_FROM><ALIGNSCORE>{align_score_values[0]}</ALIGNSCORE>\n")
            print(f"----- Finished evaluating row {i+1}/{total_bills} for {args.summary_file} summaries -----")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bill_file", default="../../preprocess/data/clean_billsum_test.csv", help="File containing bill text")
    parser.add_argument("--summary_file", default="gold", help="File containing the summary texts. Pulls gold summaries from huggingface by default")
    parser.add_argument("--output_file", default="output.txt", help="File containing the output")
    parser.add_argument("--num_examples",default=None, help="Specifies the number of examples. Evaluates all by default.")
    parser.add_argument("--checkpoint", default="https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt", help="The model checkpoint to use")
    args = parser.parse_args()

    main()