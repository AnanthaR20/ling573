from rouge_score import rouge_scorer
from readability import Readability

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
