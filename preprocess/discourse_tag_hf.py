import time
import argparse
from sample import sample_data
from transformers import pipeline
from spacy.lang.en import English

nlp = English()
nlp.add_pipe('sentencizer')
pipe = pipeline("text-classification", model="sileod/roberta-base-discourse-marker-prediction")

def discourse_tag(s):
    # this does not handle text > 512 chars
    text = s.replace("\n", "")
    doc = nlp(text)

    sent_labels = []

    for sent in doc.sents:
        if len(sent.text) > 512:
            sent_labels.append({"label": "NAN", "score": "NAN"})
        else:
            result = pipe(sent.text)
            sent_labels.append(result[0])
    return sent_labels

def tag_data(column, n=100):
    # 100 rows of data
    data = sample_data(n)
    
    data[f"{column}_tag"] = data[f"{column}_clean"].apply(discourse_tag)
    data[[f"{column}_clean", f"{column}_tag"]].to_csv("discourse_tag_data.csv", index=False)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="Returns summary column and rephrased summary column as CSV")
    parser.add_argument("--text", action="store_true", help="Returns text column and rephrased text column as CSV")
    parser.add_argument("--sample_size", default=100, type=int, help="Manually set sample size")
    args = parser.parse_args()
    start_time = time.time()
    if args.summary:
        tag_data("summary", args.sample_size)
    
    if args.text:
        tag_data("text", args.sample_size)

    print(f"Total generation time: {time.time() - start_time}")
    return 

if __name__ == "__main__":
    main()