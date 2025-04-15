from rouge_score import rouge_scorer

def get_rouge_scores(gold: str, test: str) -> dict:
    """
    Gets the ROUGE score between the gold and test strings using the rouge_score library.
    Returns a dictionary with the ROUGE scores for each metric.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    score = scorer.score(gold, test)
    return score
