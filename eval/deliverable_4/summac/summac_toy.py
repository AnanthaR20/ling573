import nltk
from summac.model_summac import SummaCZS
from summac.model_summac import SummaCConv
import pandas as pd
from datasets import load_dataset
import argparse


nltk.download('punkt_tab')

#parse command-line args
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, required=True)
parser.add_argument("--output_csv", type=str, required=True)
args = parser.parse_args()


#initializing SummaCConv model
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", agg="mean")

#load gold data and baseline
gold_sum = load_dataset("FiscalNote/billsum")["test"].to_pandas()["summary"].tolist()
gold_doc = load_dataset("FiscalNote/billsum")["test"].to_pandas()["text"].tolist()

baseline = pd.read_csv(args.input_csv)
generated_sum=baseline["predicted_summary"].tolist()

results = []

for i in range(len(gold_doc)):
    #gold document and gen summary summac score
    gen_score = model_conv.score([gold_doc[i]], [generated_sum[i]])
    print(gen_score) #keep track in case

    results.append({
        "document": gold_doc[i],
        "gold_summary": gold_sum[i],
        "generated_summary": generated_sum[i],
        "generated_score": gen_score
    })

results_df = pd.DataFrame(results)
results_df.to_csv(args.output_csv)
