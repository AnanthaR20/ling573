import spacy
import lftk
from rouge_score import rouge_scorer
import pandas as pd
from evaluate import load
from summac.model_summac import SummaCConv
import torch
from alignscore import AlignScore
from summac.model_summac import SummaCConv
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
import nltk
# globals

# load device
curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load a trained pipeline of your choice from spacy
nlp = spacy.load("en_core_web_sm")

# load NLTK corpora
# nltk.download("stopwords")
# nltk.download("punkt")

# LFTK feature families of interest
READFORMULA = lftk.search_features(family="readformula", return_format = "list_key")
WORDSENT = lftk.search_features(family="wordsent", return_format = "list_key")
WORDDIFF = lftk.search_features(family="worddiff", return_format = "list_key")
ALL_FEATURES = READFORMULA + WORDSENT + WORDDIFF

# load scoring objects
# AlignScore model
align_scorer = AlignScore(model='roberta-base', batch_size=32, device=curr_device, ckpt_path='AlignScore-base.ckpt', evaluation_mode='nli_sp')
# BERTScore scorer; returns P, R, F1
bertscore = load("bertscore")

# ROUGE - defining types to define scorer
rouge_types = ["rouge1","rouge2","rougeL","rougeLsum"]
score_types = ["precision", "recall", "f1"]
rouge = rouge_scorer.RougeScorer(rouge_types=rouge_types,use_stemmer=True)

# SummaC
summac_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=curr_device, agg="mean")
# Manually set SummaC model into evaluation mode
summac_conv.eval()

def eval_alignscore_batch(batch):
  """
  Get AlignScore for a Dataset batch

  Arguments: 
    batch: dict with keys 'text' and 'predicted_summary'
  Returns:
    dict with 'align_score' list aligned with batch
  """
  return {
    "align_score": align_scorer.score(contexts=batch["text"], claims=batch["predicted_summary"])
  }

def eval_alignscore_tmp(texts, summaries):
  return {
    "align_score": align_scorer.score(contexts=texts, claims=summaries)
  }

def eval_summac_batch(texts, summaries):
  """
  Get SummaC score for a Dataset batch

  Arguments: 
    batch: dict with keys 'text' and 'predicted_summary'
  Returns:
    dict with 'summac_score' list aligned with batch
  """
  scores = None
  with torch.no_grad():
    # Make sure that no gradients are computed just in case that matters
    scores = summac_conv.score(texts, summaries)["scores"]
  assert scores is not None
  return {
    "summac": scores
  }

# def eval_summac(bill_text:str,gen_text:str) -> dict[str, float|int]:
#   """Gets the precision, recall, and fmeasure scores for summac

#   Arguments:
#     sum_text: the gold summary for a bill
#     gen_text: the model generated summary for the same bill
  
#   Returns:
#     a dictionary where key = summac and value = the score
#   """
#   summac = summac_conv.score([bill_text], [gen_text])

def eval_bert(gold_text:str,gen_text:str) -> dict[str, float|int]:
  """Gets the precision, recall, and fmeasure scores for bert

  Arguments:
    gold_text: the gold summary for a bill
    gen_text: the model generated summary for the same bill
  
  Returns:
    a dictionary where key = bert metric type and value = the score
  """
  
  prediction = [gen_text]
  reference = [gold_text]
  bert_results = bertscore.compute(predictions=prediction, references=reference, model_type="distilbert-base-uncased")
  results = dict()
  for s in score_types:
    curr = bert_results[s]
    results[f"bert_score_{s}"] = sum(curr)/len(curr)
  return results

def eval_rouge_batch(batch):
  """
  Return a list of ROUGE score dicts corresponding to each example in a batch
  """
  # Store ROUGE scores in a dict of lists
  results = {}
  for rt in rouge_types:
    results[f"{rt}_precision"] = []
    results[f"{rt}_recall"] = []
    results[f"{rt}_fmeasure"] = []

  for ref, pred in zip(batch["summary"], batch["predicted_summary"]):
    scores = rouge.score(ref, pred)
    for rt in rouge_types:
      results[f"{rt}_precision"].append(scores[rt].precision)
      results[f"{rt}_recall"].append(scores[rt].recall)
      results[f"{rt}_fmeasure"].append(scores[rt].fmeasure)
  return results

def eval_rouge(gold_text:str,gen_text:str) -> dict[str,float|int]:
  """Gets the precision, recall, and fmeasure scores for rouge1,rouge2,rougeL, 
  and rougeLsum

  Arguments:
    gold_text: the gold summary for a bill
    gen_text: the model generated summary for the same bill
  
  Returns:
    a dictionary where key = rouge metric and type and value = the score
  """
  scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL","rougeLsum"],use_stemmer=True)
  results = scorer.score(gold_text,gen_text)

  return {
    "rouge1_precision": results['rouge1'].precision,
    "rouge2_precision": results['rouge2'].precision,
    "rougeL_precision": results['rougeL'].precision,
    "rougeLsum_precision": results['rougeLsum'].precision,
    "rouge1_recall": results['rouge1'].recall,
    "rouge2_recall": results['rouge2'].recall,
    "rougeL_recall": results['rougeL'].recall,
    "rougeLsum_recall": results['rougeLsum'].recall,
    "rouge1_fmeasure": results['rouge1'].fmeasure,
    "rouge2_fmeasure": results['rouge2'].fmeasure,
    "rougeL_fmeasure": results['rougeL'].fmeasure,
    "rougeLsum_fmeasure": results['rougeLsum'].fmeasure
  }

def eval_lftk(text:str, lftk_features:list[str] = ALL_FEATURES, suffix:str = "") -> dict[str,float]:
  """Gets the specified lftk features for a given text input.

  Arguments:
    text: The text whose LFTK properties you'd like to know
    lftk_features: A list of the desired lftk features as strings. Can get this
      from lftk.search_features(...)
    suffix: an optional parameter for adding a suffix to the key names of the
      returned dictionary

  Returns:
     a dictionary with key = lftk feature and value = float or int
  """
  LFTK = lftk.Extractor(docs = nlp(text))
  feature_dict = LFTK.extract(lftk_features)
  return {key+suffix:val for key,val in feature_dict.items()}

################################################################################
# Information coverage accuracy
################################################################################
def get_decision_metrics(
    preds, 
    targets,
    special_tokens=['<s>', '</s>', '<unk>', '<pad>', '<mask>','[SUMMARIZE]', '[NO_SUMMARY]']
    ):
  # Store counts of each classification
  counts = {
    "TP": 0,
    "FP": 0,
    "TN": 0,
    "FN": 0
  }
  # Pair up each prediction and reference (which are already sorted)
  for pred, target in zip(preds, targets):
    # Clean up special tokens from text for this process
    for special_token in special_tokens:
      pred = pred.replace(special_token, "").strip()
      target = target.replace(special_token, "").strip() # shouldn't do anything, but doesn't hurt

    # Does prediction contain text? T/F
    pred_pos = bool(len(pred)) 
    # Does reference contain text? T/F 
    ref_pos = bool(len(target))

    if ref_pos and pred_pos:
      counts["TP"] += 1
    elif not ref_pos and pred_pos:
      counts["FP"] += 1
    elif not ref_pos and not pred_pos:
      counts["TN"] += 1
    elif ref_pos and not pred_pos:
      counts["FN"] += 1
  
  print("Counts by classification results:")
  print(counts)

  accuracy = (counts["TP"] + counts["TN"])/sum(counts.values())
  precision = counts["TP"]/(counts["TP"] + counts["FP"])
  recall = counts["TP"]/(counts["TP"] + counts["FN"])
  fscore = (2 * precision * recall)/(precision + recall)

  print(f"Decision accuracy: {accuracy}")
  print(f"Decision precision: {precision}")
  print(f"Decision recall: {recall}")
  print(f"Decision F1: {fscore}")
  return

################################################################################
# Adapted from Se3
################################################################################

def get_bertscore_metrics(preds, refs):
    """Computes the bertscore metric.

    Args:
        preds: list. The model predictions.
        refs: list. The references.

    Returns:
        The bertscore metrics.
    """

    bertscore_output = bertscore.compute(predictions=preds, references=refs, lang="en")
    # return {
    #     "bertscore_p": round(np.mean([v for v in bertscore_output["precision"]]), 4),
    #     "bertscore_r": round(np.mean([v for v in bertscore_output["recall"]]), 4),
    #     "bertscore_f1": round(np.mean([v for v in bertscore_output["f1"]]), 4)
    # }
    return {
        "bertscore_p": bertscore_output["precision"],
        "bertscore_r": bertscore_output["recall"],
        "bertscore_f1": bertscore_output["f1"]
    }


def get_redundancy_scores(preds):
    sum_unigram_ratio = 0
    sum_bigram_ratio = 0
    sum_trigram_ratio = 0
    all_unigram_ratio = []
    all_bigram_ratio = []
    all_trigram_ratio = []

    sum_redundancy = 0
    stop_words = set(stopwords.words("english"))
    count = CountVectorizer()
    all_redundancy = []

    number_file = len(preds)

    for p in preds:
        all_txt = []
        all_txt.extend(word_tokenize(p.strip()))

        # uniq n-gram ratio
        all_unigram = list(ngrams(all_txt, 1))
        uniq_unigram = set(all_unigram)
        unigram_ratio = len(uniq_unigram) / len(all_unigram)
        sum_unigram_ratio += unigram_ratio

        all_bigram = list(ngrams(all_txt, 2))
        uniq_bigram = set(all_bigram)
        bigram_ratio = len(uniq_bigram) / len(all_bigram)
        sum_bigram_ratio += bigram_ratio

        all_trigram = list(ngrams(all_txt, 3))
        uniq_trigram = set(all_trigram)
        trigram_ratio = len(uniq_trigram) / len(all_trigram)
        sum_trigram_ratio += trigram_ratio

        all_unigram_ratio.append(unigram_ratio)
        all_bigram_ratio.append(bigram_ratio)
        all_trigram_ratio.append(trigram_ratio)

        # NID score
        num_word = len(all_txt)
        new_all_txt = [w for w in all_txt if not w in stop_words]
        new_all_txt = [' '.join(new_all_txt)]

        try:
            x = count.fit_transform(new_all_txt)
            bow = x.toarray()[0]
            max_possible_entropy = np.log(num_word)
            e = entropy(bow)
            redundancy = (1 - e / max_possible_entropy)
            sum_redundancy += redundancy
            all_redundancy.append(redundancy)
        except ValueError:
            continue

    print(f'Number of documents: {number_file}, average unique unigram ratio is {round(sum_unigram_ratio/number_file, 4)}, average unique bigram ratio is {round(sum_bigram_ratio/number_file, 4)}, average unique trigram ratio is {round(sum_trigram_ratio/number_file, 4)}, NID score is {round(sum_redundancy/number_file, 4)}.')
    return all_unigram_ratio, all_bigram_ratio, all_trigram_ratio, all_redundancy


def eval_all(
  gold_text: str, 
  gen_text: str,
  bill_text: str,
  include_rouge:bool = True,
  include_bert:bool = True,
  include_summac:bool = True,
  include_read_formula:bool = True,
  include_word_sent:bool = True,
  include_word_diff:bool = False,
  skip_gold_lftk:bool = True
) -> dict[str,float|int]:
  """Gets evaluation metrics for a pair of texts. Intended to receive a gold and
  generated summary and output rouge scores for the pair and lftk feature scores
  for each text- using a suffix on key name to denote lftk for generated and lftk 
  for gold. If all 'include_*' arguments set to False, will return an empty dict.

  Arguments:
    gold_data: reference/human-written summary of a bill
    gen_data: model-generated summary of the same bill
    include_rouge: whether to include the rouge family of metrics in the final dict 
    include_read_formula: whether to include the lftk 'read_formula' 
      family of metrics
    include_word_sent: whether to include the lftk 'word_sent' family of metrics
    include_word_diff: whether to include the lftk 'word_diff' family of metrics 
    skip_gold_lftk: flag to indicate if we should skip lftk metrics for 'gold_text'

  Returns:
    a dictionary of all the requested metrics for the given pair of 
    gold and generated text
  """
  results = {}

  # Do summac inclusion
  if include_summac:
    results.update(eval_summac(bill_text,gen_text))

  # Do bert inclusion
  if include_bert:
    results.update(eval_bert(gold_text,gen_text))

  # Do rouge inclusion
  if include_rouge:
    results.update(eval_rouge(gold_text,gen_text))

  # Do lftk feature inclusion
  _readformula = READFORMULA if include_read_formula else []
  _wordsent = WORDSENT if include_word_sent else []
  _worddiff = WORDDIFF if include_word_diff else []
  lftk_features = _readformula + _wordsent + _worddiff
  
  # For gold
  if not skip_gold_lftk:
    results.update(eval_lftk(gold_text,lftk_features,suffix=".GOLD"))

  # For gen
  results.update(eval_lftk(gen_text,lftk_features,suffix=".GEN"))

  print("BERT, Rouge, and LFTK evaluated")
  return results


if __name__ == "__main__":
  data = pd.read_csv("../output/deliverable_4/wugwATSS-pegasus/eval_on_simp.csv", usecols=["predicted_summary"])
  gold = pd.read_csv("gold_lftk.csv", usecols=["summary"])
  bills = pd.read_csv("gold_lftk.csv", usecols=["summary"]) #fix this so it points to bill list
  # gold = load_dataset("FiscalNote/billsum")["test"].to_pandas()
  gold_list = gold.summary.tolist()
  gen_list = data.predicted_summary.tolist()
  bill_list = gold.summary.tolist() #fix this so it points to bill list
  scores = []
  for g, p, b in zip(gold_list, gen_list, bill_list):
    scores.append(eval_all(g, p, b))
  final = pd.DataFrame(scores)
  final.to_csv("wugwATSS-pegasus_eval_with_bert_scores.csv", index=False)

