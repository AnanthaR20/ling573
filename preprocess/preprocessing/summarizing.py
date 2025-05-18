import pandas as pd
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = "google/pegasus-billsum"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

def summarize(text):
    batch = tokenizer(text, truncation=True, padding="longest", return_tensors="pt").to(device)
    output = model.generate(**batch)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

df = pd.read_csv("simplified_bills.csv")
df['summary'] = df['simplified_text'].apply(summarize)
df[['summary']].to_csv("summarized_bills.csv", index=False)
