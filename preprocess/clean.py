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

    ds = load_dataset(args.dataset)
    
    for split, dataset in ds.items():
        # Toy experiment setting
        if args.toy:
            dataset = dataset.select(range(args.toy))
        dataset = dataset.map(clean_text)
        if args.output_file:
            dataset.to_csv(f"data/{args.output_file}_{split}_clean.csv", index=None, escapechar="\\")
        dataset.to_csv(f"data/{args.dataset}_{split}_clean.csv", index=None, escapechar="\\")

if __name__ == "__main__":
    main()