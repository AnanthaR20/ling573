from datasets import load_dataset
import argparse
import re

def clean(df):
    df["text_clean"] = df.text.str.replace("\n", "").apply(lambda b: re.sub(' +', ' ', b))
    df["summary_clean"] = df.summary.str.replace("\n", "").apply(lambda b: re.sub(' +', ' ', b))
    return 

def sample_data(n=100):
    ds = load_dataset("FiscalNote/billsum")
    sample_data = ds["train"].train_test_split(test_size=n, seed=42)
    result = sample_data["test"].to_pandas()
    clean(result)
    return result

def to_text(n=100):
    sample_data(n)["text_clean"].to_csv("sample_text.txt", index=False)
    sample_data(n)["summary_clean"].to_csv("sample_summary.txt", index=False)
    return

def to_csv(n=100):
    sample_data(n)["text_clean"].to_csv("sample_text.csv", index=False)
    sample_data(n)["summary_clean"].to_csv("sample_summary.csv", index=False)
    return 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--to_txt", action="store_true", help="Returns text column and summary column as .txt files")
    parser.add_argument("--to_csv", action="store_true", help="Returns text column and summary column as .csv files")
    parser.add_argument("--sample_size", help="Manually set sample size")
    args = parser.parse_args()

    if args.to_txt:
        to_text(args.sample_size)
    elif args.to_csv:
        to_csv(args.sample_size)
    else:
        print("oh dear")

if __name__ == "__main__":
    main()