from summac.model_summac import SummaCZS, SummaCConv
from datasets import load_dataset
import pandas as pd
from collections import defaultdict

#initialize model
model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cpu") # If you have a GPU: switch to: device="cuda"
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", agg="mean")

#load our gold data
gold_sum = load_dataset("FiscalNote/billsum")["test"].to_pandas()["summary"].tolist()
gold_doc = load_dataset("FiscalNote/billsum")["test"].to_pandas()["text"].tolist()
#load generated summaries
baseline = pd.read_csv("../output/baseline_test.csv")

#initialize data structure to store values
ref_list_conv = [9999] * baseline.shape[0]
gen_list_conv = [9999] * baseline.shape[0]
ref_list_zs = [9999] * baseline.shape[0]
gen_list_zs = [9999] * baseline.shape[0]

#initialize a safety output file in case of segmentation issues
safety_output = 'summac_safety.txt'
#clear safety output:
with open(safety_output, 'w') as file:
    file.write('')
#itereate through rows and calculate scores
#for i in range(baseline.shape[0]):
for i in range(3):
    print("processing pair: " + str(i))
    doc = gold_doc[i]
    ref = gold_sum[i]
    gen = baseline.loc[i].summary_generated
    ref_score_conv = model_conv.score([doc], [ref])
    if ref_score_conv:
        ref_list_conv[i] = ref_score_conv
    gen_score_conv = model_conv.score([doc], [gen])
    if gen_score_conv:
        gen_list_conv[i] = gen_score_conv
    with open(safety_output, 'a') as file:
        file.write("ref_conv:" + str(ref_score_conv) + " gen_conv:" + str(gen_score_conv)+ '\n')
output_dict = {'ref_conv': ref_list_conv, 'gen_conv': gen_list_conv}
df = pd.DataFrame(output_dict)
df.to_csv('summac_scores.csv', index=False)
