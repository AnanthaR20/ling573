# clean_bills.py
import pandas as pd
import re
from datasets import load_dataset

SECTION_HEADER_RE = re.compile(r'\b(SEC(?:TION)?\.?\s*\d+[A-Za-z]?\.*.*?)\n+')
SUBSECTION_RE = re.compile(r'\(\w+\)\s*')
PARENTH_RE = re.compile(r'\n\s*\(\d+\)\s*')
STRIP_RE = re.compile(r'\s+')

splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet',
    'ca_test': 'data/ca_test-00000-of-00001.parquet'
}
df = pd.read_parquet("hf://datasets/FiscalNote/billsum/" + splits["test"])

def clean_text(text):
    text = SECTION_HEADER_RE.sub('', text)
    text = SUBSECTION_RE.sub('', text)
    text = PARENTH_RE.sub(' ', text)
    return STRIP_RE.sub(' ', text).strip()

df['cleaned_text'] = df['text'].apply(clean_text)

df[['cleaned_text']].to_csv("cleaned_bills.csv", index=False)
