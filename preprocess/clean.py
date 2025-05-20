import pandas as pd
import re
from datasets import load_dataset
import argparse

SECTION_HEADER_RE = re.compile(r'\b(SEC(?:TION)?\.?\s*\d+[A-Za-z]?\.*.*?)\n+')
SUBSECTION_RE = re.compile(r'\(\w+\)\s*')
PARENTH_RE = re.compile(r'\n\s*\(\d+\)\s*')
STRIP_RE = re.compile(r'\s+')

def clean_text(example):
    example["text"] = SECTION_HEADER_RE.sub('', example["text"])
    example["text"] = SUBSECTION_RE.sub('', example["text"])
    example["text"] = PARENTH_RE.sub(' ', example["text"])
    example["text"] = STRIP_RE.sub(' ', example["text"]).strip()
    return example

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="billsum", help="specify HuggingFace dataset")
    parser.add_argument("--output_file",default=None,help="overrides default naming schema")
    parser.add_argument("--toy", default=0, type=int, help="specify size of toy datset to clean")
    args = parser.parse_args()

    # Preload output filename
    outname = f"data/{args.dataset}_clean.csv"
    if args.output_file:
        print("Overriding default naming schema...")
        outname = args.output_file

    ds = load_dataset(args.dataset)
    
    for split, dataset in ds.items():
        curr_name = outname.split(".")[0] + f"_{split}.csv"
        # Toy experiment setting
        if args.toy:
            curr_name = curr_name.split(".")[0] + "_toy.csv"
            dataset = dataset.select(range(args.toy))
        dataset = dataset.map(clean_text)
        dataset[:, ["text", "summary"]].to_csv(curr_name, index=None, escapechar="\\")

if __name__ == "__main__":
    main()