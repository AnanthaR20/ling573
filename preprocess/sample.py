from datasets import load_dataset, train_test_split
import argparse

def sample(n=100):
    ds = load_dataset("FiscalNote/billsum")
    sample_data = ds.train_test_split(test_size=n, seed=42)["test"]
    return sample_data.to_pandas()

def to_text():
    sample()["text"].to_csv("sample_text.txt", header=False, index=False)
    sample()["summary"].to_csv("sample_summary.txt", header=False, index=False)
    return

def to_csv():
    sample()["text"].to_csv("sample_text.csv", header=False, index=False)
    sample()["summary"].to_csv("sample_summary.csv", header=False, index=False)
    return 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--to_txt", help="Returns text column and summary column as .txt files")
    parser.add_argument("--to_csv", help="Returns text column and summary column as .csv files")
    args = parser.parse_args()

    if args.to_txt:
        to_text()
    elif args.to_csv:
        to_csv()
    else:
        print("oh dear")