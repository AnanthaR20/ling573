from rouge_score import rouge_scorer
import pandas as pd
#from readability import Readability


def get_rouge_scores(gold: str, test: str) -> dict:
    """
    Gets the ROUGE score between the gold and test strings using the rouge_score library.
    Returns a dictionary with the ROUGE scores for each metric.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    score = scorer.score(gold, test)
    return score

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


"""
Looking at the test split from HuggingFace for gold summaries
"""

splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'ca_test': 'data/ca_test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/FiscalNote/billsum/" + splits["test"])
gold = []
for d in df['summary']:
    gold.append(d)

"""
Access generated summaries from pegasus-billsum model 
"""
generated_summaries = pd.read_csv('baseline_test.csv', usecols=['summary_generated'])


print(f'rouge1\trouge2\trougeL')

r1prec_sum = 0
r1recall_sum = 0
r1f_sum = 0
r2prec_sum = 0
r2recall_sum = 0
r2f_sum = 0
rlprec_sum = 0
rlrecall_sum = 0
rlf_sum = 0

for g, s in zip(gold, generated_summaries['summary_generated']):
    scores = get_rouge_scores(g, s)
    r1 = scores['rouge1']
    r2 = scores['rouge2']
    rl = scores['rougeL']

    r1_precision = scores['rouge1'][0]
    r1prec_sum+=r1_precision
    r2_precision = scores['rouge2'][0]
    r2prec_sum+=r2_precision
    rl_precision = scores['rougeL'][0]
    rlprec_sum+=rl_precision
    r1_recall = scores['rouge1'][1]
    r1recall_sum += r1_recall
    r2_recall = scores['rouge2'][1]
    r2recall_sum +=r2_recall
    rl_recall = scores['rougeL'][1]
    rlrecall_sum += rl_recall
    r1_fmeasure = scores['rouge1'][2]
    r1f_sum += r1_fmeasure
    r2_fmeasure = scores['rouge2'][2]
    r2f_sum += r2_fmeasure
    rl_fmeasure = scores['rougeL'][2]
    rlf_sum += rl_fmeasure

    print(f'{r1}\t{r2}\t{rl}')

print('\n\n\n')

print(f'AVERAGES\nrouge1 precision: {r1prec_sum/len(gold)}\n'
      f'rouge1 recall: {r1recall_sum/len(gold)}\n'
      f'rouge1 fmeasure: {r1f_sum/len(gold)}\n'
      f'rouge2 precision: {r2prec_sum/len(gold)}\n'
      f'rouge2 recall: {r2recall_sum/len(gold)}\n'
      f'rouge2 fmeasure: {r2f_sum/len(gold)}\n'
      f'rougel precision: {rlprec_sum/len(gold)}\n'
      f'rougel recall: {rlrecall_sum/len(gold)}\n'
      f'rougel fmeasure: {rlf_sum/len(gold)}\n'
      )


