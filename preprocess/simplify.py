import pandas as pd
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
import argparse

checkpoint = "unikei/t5-base-split-and-rephrase"
tokenizer = T5Tokenizer.from_pretrained(checkpoint, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

def simplify_bill(example, idx, max_input_len, max_output_len):
    try: 
        # Tokenize input
        tokens = tokenizer(
            example["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_input_len, 
            return_tensors='pt'
        )
        # Generate output tokens with suggested params
        output_ids = model.generate(
            tokens['input_ids'], 
            attention_mask=tokens['attention_mask'], 
            max_length=max_output_len, 
            num_beams=5
        )
        # 
        # Decode the result
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # Overwrite the original text column with decoded simple text
        example["text"] = decoded
    except IndexError:
        print(f"IndexError at index {idx}")
        print("First 100 characters so you can look into it manually")
        print(example["text"][:100])
    return example

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
    args = parser.parse_args()

    # Preload output name
    outname = f"data/{args.dataset}_clean_{args.split}_{args.chunk_type}_simple.csv"
    if args.output_file:
        print("Overriding default naming schema...")
        outname = args.output_file

    # Determine source name by arguments
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

    # Map into function - N rows will return N rows
    ds = ds.map(
        simplify_bill, 
        batched=True,
        batch_size=25,
        with_indices=True,
        fn_kwargs={
            "max_input_len": args.max_input_len,
            "max_output_len": args.max_output_len
        }
    )
    # Write to output
    ds.to_csv(outname, index=None, escapechar="\\")
    return

if __name__ == "__main__":
    main()
