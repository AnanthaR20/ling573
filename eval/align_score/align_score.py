"""
This script is for generating alignscore factuality metrics.

Last Updated: 7-18-2025
"""
import argparse
import pandas as pd
from alignscore import AlignScore
# from datasets import load_dataset
# df = load_dataset("FiscalNote/billsum")
# docs = df['test'].to_pandas()['text']
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
    # Assemble the files into corresponding lists
    bills = []
    generated_summaries = []
    args.num_examples = int(args.num_examples)

    if args.num_examples:
        with open(args.bill_file,'r') as f:
            bills = f.read().splitlines()[1:args.num_examples+1]
        generated_summaries = list(pd.read_csv(args.summary_file)["predicted_summary"])[0:args.num_examples]
    else:
        with open(args.bill_file,'r') as f:
            bills = f.read().splitlines()[1:]
        generated_summaries = list(pd.read_csv(args.summary_file)["predicted_summary"])

    # Iteratively record AlignScore values in case it quits randomly at a certain point.
    for i,context,claim in zip(range(len(bills)),bills,generated_summaries):
        align_score_values = eval_alignscore([context],[claim])
        print(f"<ROW>{i}</ROW><BILL>{context}</BILL><GENERATED_SUMMARY>{claim}</GENERATED_SUMMARY><ALIGNSCORE>{align_score_values[0]}</ALIGNSCORE>")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bill_file", default="../../preprocess/data/clean_billsum_test.csv", help="File containing bill text")
    parser.add_argument("--summary_file", default="../../output/deliverable_4/led-base/led-base_billsum_clean_test_se3-led-2048-512.csv", help="File containing the summary texts")
    parser.add_argument("--num_examples",default=None, help="Specifies the number of examples. Evaluates all by default.")
    parser.add_argument("--checkpoint", default="https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt", help="The model checkpoint to use")
    args = parser.parse_args()

    main()