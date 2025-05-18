import pandas as pd
from rouge_score import rouge_scorer, scoring
from readability import Readability


#######################################
# ------- GET SCORES------------------#
#######################################
#ROUGE SCORES
def get_rouge_scores(gold: str, test: str) -> dict:
    """
    Gets the ROUGE score between the gold and test strings using the rouge_score library.
    Returns a dictionary with the ROUGE scores for each metric.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    score = scorer.score(gold, test)
    return score

#READABILITY SCORES
def get_readability_scores(text: str) -> dict:
    """
    Gets a dictionary of readability scores and other metadata for a given string after tokenizing.
    """
    r = Readability(text)
    return {
        "flesch_kincaid": r.flesch_kincaid(),
        "flesch": r.flesch(),
        "gunning_fog": r.gunning_fog(),
        "coleman_liau": r.coleman_liau(),
        "dale_chall": r.dale_chall(),
        "ari": r.ari(),
        "linsear_write": r.linsear_write(),
        "smog": r.smog(),
        "spache": r.spache()
    }


#######################################
# ------- LOAD DATA-------------------#
#######################################
splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet',
    'ca_test': 'data/ca_test-00000-of-00001.parquet'
}
df = pd.read_parquet("hf://datasets/FiscalNote/billsum/" + splits["test"])
generated_summaries = pd.read_csv('../output/baseline_test.csv', usecols=['summary_generated'])

#######################################
# ------- GET AGGREGATORS-------------#
#######################################
rouge_aggregators = {
    'rouge1': scoring.BootstrapAggregator(),
    'rouge2': scoring.BootstrapAggregator(),
    'rougeL': scoring.BootstrapAggregator()
}
readability_all = []


#######################################
# -------EVALUATE---------------------#
#######################################

for gold, gen in zip(df['summary'], generated_summaries['summary_generated']):
    # ROUGE
    rouge_scores = get_rouge_scores(gold, gen)
    for key in rouge_aggregators:
        rouge_aggregators[key].add_scores(rouge_scores[key])

    # Readability
    readability_all.append(get_readability_scores(gen))


#######################################
# ------- GET OUTPUT------------------#
#######################################
print("\nAVERAGE ROUGE SCORES")
for key, aggregator in rouge_aggregators.items():
    result = aggregator.aggregate()
    print(f"{key}: precision={result['precision']['mid']}, recall={result['recall']['mid']}, f1={result['fmeasure']['mid']}")

print("\nAVERAGE READABILITY SCORES")
n = len(readability_all)
readability_totals = {}
for key, total in readability_totals.items():
    print(f"{key}: {total / n}")