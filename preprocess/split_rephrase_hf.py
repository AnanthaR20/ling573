import argparse
import time
from transformers import pipeline
from sample import sample_data

pipe = pipeline("text2text-generation", model="unikei/t5-base-split-and-rephrase")

def split_rephrase(row):
    # this does not handle text > 256 chars lol
    result = pipe(row.replace("\n", ""))
    return result[0]["generated_text"]

def split_data(column, n=100):
    # 100 rows of data
    data = sample_data(n)
    
    data[f"{column}_rephrase"] = data[f"{column}_clean"].apply(split_rephrase)
    data[[f"{column}_clean", f"{column}_rephrase"]].to_csv("split_rephrase_data.csv", index=False)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="Returns summary column and rephrased summary column as CSV")
    parser.add_argument("--text", action="store_true", help="Returns text column and rephrased text column as CSV")
    parser.add_argument("--sample_size", type=int, default=100, help="Manually set sample size")
    args = parser.parse_args()

    start_time = time.time()
    if args.summary:
        split_data("summary", args.sample_size)
    if args.text:
        split_data("text", args.sample_size)

    print(f"Total generation time: {time.time() - start_time}")
    return 

if __name__ == "__main__":
    main()