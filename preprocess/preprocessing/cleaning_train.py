# clean_bills.py
import pandas as pd
import re
from datasets import load_dataset

SECTION_HEADER_RE = re.compile(r'\b(SEC(?:TION)?\.?\s*\d+[A-Za-z]?\.*.*?)\n+')
SUBSECTION_RE = re.compile(r'\(\w+\)\s*')
PARENTH_RE = re.compile(r'\n\s*\(\d+\)\s*')
STRIP_RE = re.compile(r'\s+')

ds = load_dataset("FiscalNote/billsum/", split="train")

def clean_text(text):
    text = SECTION_HEADER_RE.sub('', text)
    text = SUBSECTION_RE.sub('', text)
    text = PARENTH_RE.sub(' ', text)
    return STRIP_RE.sub(' ', text).strip()

ds['cleaned_text'] = df['text'].apply(clean_text)

ds[['cleaned_text']].to_csv("cleaned_bills.csv", index=False)
