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
    parser.add_argument("--dataset", default="FiscalNote/billsum", help="specify HuggingFace dataset")
    parser.add_argument("--output_file",default=None,help="overrides default naming schema")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split = ["test", "train"])
    ds = ds.map(clean_text, batched=True)
    
    for split, dataset in ds.items():
        dataset.to_csv(f"data/{args.dataset}_{split}_clean.csv", index=None)

if __name__ == "__main__":
    main()