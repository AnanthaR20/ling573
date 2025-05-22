"""
Script for fixed or semantic (se3-based) chunking of the billsum training data
#TODO - written 5.18.2025
I'm working on getting this script to accept a file as an input rather than a string
The code from this is being pieced together from preprocessing/simplify.py and se3/se3/segmentation.py
"""
# Add hyak path
import sys
sys.path.insert(0,"/gscratch/scrubbed/jcmw614/ling573/preprocess/se3")
import argparse
import pandas as pd
import re
import pysbd
from se3 import segmentation 
from datasets import Dataset
from rouge_score import rouge_scorer

SPLIT_RE = re.compile(r'(?<=\.)\s+')
sent_tokenizer = pysbd.Segmenter(language="en", clean=True)
scorer_rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

def fixed_target(chunks, summary):
    if len(chunks) == 1:
        return [summary]
    else:
        targets = [""] * len(chunks)  # Initialize empty targets
        summary_sents = sent_tokenizer.segment(summary)
        for sent in summary_sents:
            scores = [segmentation.get_rouge1_precision(scorer_rouge, chunk, sent) for chunk in chunks]
            # Find and sort the chunks with the highest similarity score.
            indices = [i[0] for i in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
            # Update targets
            targets[indices[0]] += " " + sent
    return targets

def fixed_chunk(example, size):
    # Store variables
    chunked_text = []
    current = ""
    # Split and strip parts
    print(type(example["text"]))
    text = str(example["text"])
    full_summary = str(example["summary"])
    raw_parts = re.split(SPLIT_RE, text)
    stripped_parts = list(map(lambda s: s.strip(), raw_parts))
    # Loop
    for part in stripped_parts:
        # Update current pointer if it fits 
        if len(current) + len(part) + 1 <= size:
            current += " " + part if current else part
        else:
            # If the current pointer has content, update sentences
            if current:
                chunked_text.append(current.strip())
            
            # Update the current pointer with the stripped part if it fits
            if len(part) <= size:
                current = part
            
            # Find a cutoff if the stripped part doesn't fit in the fixed chunk size
            else:
                while len(part) > size:
                    cutoff = part.rfind(" ", 0, size)
                    cutoff = cutoff if cutoff != -1 else size
                    chunked_text.append(part[:cutoff].strip())
                    part = part[cutoff:].strip()
                # Update current pointer before looping
                current = part
    # Final update from the current pointer
    if current:
        chunked_text.append(current.strip())
    
    # Assign targets to chunks
    chunked_summary = fixed_target(chunked_text, full_summary)
    example["text_chunks"] = chunked_text
    example["summary_chunks"] = chunked_summary
    return example


def se3_chunk(text):
    #TODO: write this function using se3.se3.segmentation methods
    pass


def main():
    parser = argparse.ArgumentParser()
    # Source arguments
    parser.add_argument("--dataset", default="billsum", help="specify Huggingface dataset")
    parser.add_argument("--split", default="train", help="specify dataset partition")
    parser.add_argument("--toy", default=0, type=int, help="specify size of toy dataset to chunk")
    # Chunking arguments
    parser.add_argument("--type",default="fixed",help="Method by which we chunk the input text. Options are: fixed or se3")
    parser.add_argument("--fixed_chunk_size",default=200,help="specify size of the fixed chunks (only used when --type='fixed')")
    parser.add_argument("--max_input_len", default=1024, help="se3 chunking: specify max input length in tokens")
    parser.add_argument("--max_output_len", default=512, help="se3 chunking: specify max output length in tokens")
    # Output arguments
    parser.add_argument("--output_file",default=None,help="overrides default output naming schema")
    args = parser.parse_args()

    # Preload output filename
    outname = f"data/{args.dataset}_clean_{args.split}_{args.type}.csv"
    if args.output_file:
        print("Overriding default naming schema...")
        outname = args.output_file

    # Load CSV as pandas dataframe
    df = pd.read_csv(f"data/{args.dataset}_clean_{args.split}.csv")
    # Cast as Dataset to leverage faster processing
    ds = Dataset.from_pandas(df)
    # For toy experiments
    if args.toy:
        outname = outname.split(".")[0] + "_toy.csv"
        ds = ds.select(range(args.toy))

    if args.type == "fixed":
        # Create a stack of chunked text and summary rows from the original
        # Enable batching and remove old columns to maintain proper data shape
        chunked_ds = ds.map(
            fixed_chunk, 
            batched=True,
            batch_size=5,
            remove_columns=ds.column_names,
            fn_kwargs={
                "size": args.fixed_chunk_size,
            }
        )
        # Rename chunked columns by mapping new nones and removing old named columns
        ds_final = chunked_ds.map(lambda example: {"text": example["text_chunks"], "summary": example["summary_chunks"]}, remove_columns=["text_chunks", "summary_chunks"])
    elif args.type == "se3":
        pass # TODO: implement with segementation package (low priority)

    # Write to output
    ds_final.to_csv(outname, index=None, escapechar="\\")
    return

if __name__ == "__main__":
    main()






