import pandas as pd
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
import argparse
from ..utils.utils import *

checkpoint = "unikei/t5-base-split-and-rephrase"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="billsum", help="specify Huggingface dataset")
    parser.add_argument("--split", default="train", help="specify dataset partition")
    parser.add_argument("--chunk_type", default="fixed", help="specify chunking strategy")
    parser.add_argument("--checkpoint", default=None, help="specify checkpoint for se3 chunking strategy")
    parser.add_argument("--max_input_len", default=512, help="specify max input length in tokens")
    parser.add_argument("--max_output_len", default=512, help="specify max output length in tokens")
    parser.add_argument("--toy", default=0, type=int, help="specify size of toy dataset to simplify")
    parser.add_argument("--from_toy", default=False, action="store_true", help="use toy chunked dataset")
    parser.add_argument("--output_file",default=None,help="overrides default output naming schema")
    parser.add_argument("--input_file",default=None, help="Override default input naming schema")
    args = parser.parse_args()

    # Preload output name
    outname = f"data/{args.dataset}_clean_{args.split}_{args.chunk_type}_simple.csv"
    if args.output_file:
        print("Overriding default naming schema...")
        outname = args.output_file

    # Determine source name by arguments
    if args.input_file != None:
        sourcename = args.input_file
    else:
        sourcename = f"data/{args.dataset}_clean_{args.split}_{args.chunk_type}.csv"
    
    # Specify checkpoint if this is an se3 chunked file
    if args.chunk_type == "se3" and args.checkpoint != None:
        sourcename = sourcename.split(".")[0] + f"-{args.checkpoint}-{args.max_input_len}-{args.max_output_len}.csv"
        print("Loading se3 chunked data")
    # Check for toy file if relevant
    if args.from_toy:
        print("Loading toy file")
        sourcename = sourcename.split(".")[0] + "_toy.csv"
    # Load CSV as pandas dataframe
    df = pd.read_csv(sourcename)
    
    # Cast as Dataset to leverage faster processing
    ds = Dataset.from_pandas(df)
    
    # For toy experiments
    if args.toy:
        print("Running toy experiment")
        outname = outname.split(".")[0] + "_toy.csv"
        ds = ds.select(range(args.toy))

    # Configure model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(checkpoint, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)

    # Warm up GPU
    warm_up(model, tokenizer)

    # Map into function - N rows will return N rows
    simplify_bill = create_simplify(
        model,
        tokenizer,
        args.max_input_len,
        args.max_output_len
    )

    ds = ds.map(
        simplify_bill, 
        batched=True,
        batch_size=8,
    )
    # Write to output
    ds.to_csv(outname, index=None, escapechar="\\")
    return

if __name__ == "__main__":
    main()
