import pandas as pd
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

checkpoint = "unikei/t5-base-split-and-rephrase"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

MAX_LEN = 200

def chunk_text(text):
    sentences = []
    current = ""
    raw_parts = re.split(r'(?<=\.)\s+', text)
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if len(current) + len(part) + 1 <= MAX_LEN:
            current += " " + part if current else part
        else:
            if current:
                sentences.append(current.strip())
            if len(part) <= MAX_LEN:
                current = part
            else:
                while len(part) > MAX_LEN:
                    cutoff = part.rfind(" ", 0, MAX_LEN)
                    cutoff = cutoff if cutoff != -1 else MAX_LEN
                    sentences.append(part[:cutoff].strip())
                    part = part[cutoff:].strip()
                current = part
    if current:
        sentences.append(current.strip())
    return sentences

def simplify_bill(text):
    sentences = chunk_text(text)
    all_simple_sentences = []
    for sent in sentences:
        tokens = tokenizer(sent, padding="max_length", truncation=True, max_length=256, return_tensors='pt')
        output_ids = model.generate(tokens['input_ids'], attention_mask=tokens['attention_mask'], max_length=256, num_beams=5)
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        all_simple_sentences.extend(decoded)
    return ' '.join(all_simple_sentences)

df = pd.read_csv("cleaned_bills.csv", nrows=2)
df['simplified_text'] = df['cleaned_text'].apply(simplify_bill)
df[['simplified_text']].to_csv("simplified_bills.csv", index=False)
