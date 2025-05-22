from rouge_score import rouge_scorer
from rouge_score import scoring
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset


# def eval_for_jessica(
#   gen:str,
#   gold:str,
#   include_rouge: True,
#   include_read_formula: True,
#   include_word_sent: True,
#   include_AOA: False
# ) -> dict[str,float]
# rouge1,rouge2,rougeL,rougeLsum
# fkre,fkgl,fogi,auto
# wordsent family
# don't include t_kup,t_bry,t_subtlex but re-add-able later

# globals
PATH_TO_MODEL_OUTPUT = "../output/deliverable_2/pegasusbillsum_baseline.csv"
_ROUGE_METRIC = "rouge"

def eval_all(gold_data: list, gen_data: list) -> None:
    eval_rouge(gold_data, gen_data)
    print("ROUGE evaluated")
    return

def eval_rouge(gold_data: list, gen_data: list) -> None:
    """
    - Evaluate ROUGE scores on gold and generated summaries
    - Aggregate confidence intervals on ROUGE scores
    - Write to output
    """
    scorers_dict = {}
    scorers_dict[_ROUGE_METRIC] = rouge_scorer.RougeScorer(
      ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    aggregators_dict = {k: scoring.BootstrapAggregator() for k in scorers_dict}

    for gold, gen in tqdm(zip(gold_data, gen_data)):
        for key, scorer in scorers_dict.items():
            scores_i = scorer.score(gold, gen)
            aggregators_dict[key].add_scores(scores_i)
    
    aggregates_dict = {k: v.aggregate() for k, v in aggregators_dict.items()}

    rouge_filename = "rouge_scores.txt"
    with open(rouge_filename, "w") as f:
        for k, v in sorted(aggregates_dict[_ROUGE_METRIC].items()):
            f.write("%s-R,%f,%f,%f\n" %
              (k, v.low.recall, v.mid.recall, v.high.recall))
            f.write("%s-P,%f,%f,%f\n" %
              (k, v.low.precision, v.mid.precision, v.high.precision))
            f.write("%s-F,%f,%f,%f\n" %
              (k, v.low.fmeasure, v.mid.fmeasure, v.high.fmeasure))
    return

def get_factuality_scores(text: str) -> dict:
    """
    Gets a dictionary of factuality scores for a summary.
    """
    return None

if __name__ == "__main__":
    data = pd.read_csv(PATH_TO_MODEL_OUTPUT, usecols=["summary_generated"])
    gold = load_dataset("FiscalNote/billsum")["test"].to_pandas()
    temp_gold = gold.summary.tolist()
    temp_gen = data.summary_generated.tolist()
    eval_all(temp_gold, temp_gen)
