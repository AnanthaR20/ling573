from transformers import T5Tokenizer, T5ForConditionalGeneration
checkpoint="unikei/t5-base-split-and-rephrase"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)
import pandas as pd
import re

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'ca_test': 'data/ca_test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/FiscalNote/billsum/" + splits["test"])

for d in df['text'][0:2]:

    d = re.sub(r'\b(SEC(?:TION)?\.?\s*\d+[A-Za-z]?\.*.*?)\n+', '', d, flags=re.IGNORECASE) #remove section headers
    d = re.sub(r'\(\w+\)\s*', '', d) #remove subsection headers
    d = re.sub(r'\n\s*\(\d+\)\s*', ' ', d) #remove () markers 
    d = re.sub(r'\s+', ' ', d).strip() #remove whitespace and lines

    sentences = []
    current = ""
    max_len = 200

    raw_parts = re.split(r'(?<=\.)\s+', d)

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if len(current) + len(part) + 1 <= max_len:
            if current:
                current += " " + part
            else:
                current = part
        else:
            if current:
                sentences.append(current.strip())
            if len(part) <= max_len:
                current = part
            else:
                while len(part) > max_len:
                    cutoff = part.rfind(" ", 0, max_len)
                    if cutoff == -1:
                        cutoff = max_len
                    sentences.append(part[:cutoff].strip())
                    part = part[cutoff:].strip()
                current = part
    if current:
        sentences.append(current.strip())

    for complex_sentence in sentences:
        complex_tokenized = tokenizer(complex_sentence, 
                                        padding="max_length", 
                                        truncation=True,
                                        max_length=256, 
                                        return_tensors='pt')

        simple_tokenized = model.generate(complex_tokenized['input_ids'], attention_mask = complex_tokenized['attention_mask'], max_length=256, num_beams=5)
        simple_sentences = tokenizer.batch_decode(simple_tokenized, skip_special_tokens=True)
        print(''.join(simple_sentences))
