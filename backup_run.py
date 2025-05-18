import torch
import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_dataset

# Try loading data from parquets to see if that introduces problems
splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'ca_test': 'data/ca_test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/FiscalNote/billsum/" + splits["test"])

# I ran this on my terminal for 30 hours but ideally this would have been directly writing to a file
def generate():
    # Initialize Model
    model_name = "google/pegasus-billsum"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    # Generate summaries
    for i, doc in enumerate(df['text']):
        batch = tokenizer(doc, truncation=True, padding="longest", return_tensors="pt").to(device)
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        print(tgt_text)

# Post-hoc processing - align generated summaries with titles 
def align():
    data = pd.read_csv("output/baseline_console.txt", delimiter="\t", dtype=str)
    data["title"] = df.title
    data.to_csv("output/baseline_test.csv", index=False)
    return

def main():
    align()

if __name__ == "__main__":
    main()