import pandas as pd
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
import argparse

checkpoint = "unikei/t5-base-split-and-rephrase"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

def simplify_bill(example, max_input_len, max_output_len):
    # Tokenize input
    tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_input_len, return_tensors='pt')
    # Generate output tokens with suggested params
    output_ids = model.generate(tokens['input_ids'], attention_mask=tokens['attention_mask'], max_length=max_output_len, num_beams=5)
    # Decode the result
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # Overwrite the original text column with decoded simple text
    example["text"] = decoded
    return example

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="billsum", help="specify Huggingface dataset")
    parser.add_argument("--split", default="train", help="specify dataset partition")
    parser.add_argument("--chunk_type", default="fixed_200", help="specify chunking strategy")
    parser.add_argument("--max_input_len", default=256, help="specify max input length in tokens")
    parser.add_argument("--max_output_len", default=256, help="specify max output length in tokens")
    parser.add_argument("--output_file",default=None,help="overrides default output naming schema")
    args = parser.parse_args()

    # Load CSV as pandas dataframe
    df = pd.read_csv(f"data/{args.dataset}_{args.split}_clean_{args.chunk_type}.csv")
    # Cast as Dataset to leverage faster processing
    ds = Dataset.from_pandas(df)
    # Map into function - N rows will return N rows
    ds = ds.map(
        simplify_bill, 
        batched=True, 
        fn_kwargs={
            "max_input_len": args.max_input_len,
            "max_output_len": args.max_output_len
        }
    )
    # Write to output
    if args.output_file:
        ds.to_csv(args.output_file, index=None, escapechar="\\")
    ds.to_csv(f"data/{args.dataset}_{args.split}_clean_{args.chunk_type}_simple.csv", index=None, escapechar="\\")
    return

if __name__ == "__main__":
    main()
