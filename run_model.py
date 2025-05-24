import sys
sys.path.insert(0,"/gscratch/scrubbed/jcmw614/ling573/preprocess/se3") # TODO: make this nicer
import argparse
import numpy as np
import os
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from eval.eval_metrics import eval_all
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, set_seed
# from preprocess.se3.se3.segmentation import get_rouge1_precision
from tqdm.auto import tqdm
# from preprocess import clean, chunk, simplify # TODO: figure out if this is legal, given the existence of another official preprocess pkg

def reconstruct(preds, data_index):
    final_summaries = []
    rows_seen = 0

    for i in data_index: 
        chunk_count = int(i)
        summary = ""
        for j in range(chunk_count):
            summary += preds["prediction"][rows_seen + j].strip()
        final_summaries.append(summary)
        rows_seen += chunk_count # update the number of rows seen
    return final_summaries

def create_prediction(max_input_len, max_output_len, tokenizer, model):
    def predict(examples):
        inputs = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_input_len, 
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_output_len,
            num_beams=2
        )
        return {
            "prediction": tokenizer.batch_decode(outputs, skip_special_tokens=True)
        }
    return predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="google/pegasus-billsum", help="The model checkpoint to use")
    parser.add_argument("--testfile", default="", help="The test filepath to use")
    # parser.add_argument("--fulltext", default=False, action="store_true", help="To not use the chunking")
    parser.add_argument("--concat", default="pre", help="Specify when to concatenate chunks")
    parser.add_argument("--mode", default="predict", help="To specify prediction or pipeline mode")
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size for generating predictions")
    parser.add_argument("--device", default="cuda", help="The device to use") # TODO: how to add to config file?
    # parser.add_argument("--max_input_len", type=int, default=2048, help="The input max size")
    # parser.add_argument("--max_output_len", type=int, default=512, help="The output max size")    
    args = parser.parse_args()

    # Set random seed
    set_seed(1234)
    # set_seed(args.seed)
    # has_global_attn = args.checkpoint == "allenai/led-base-16384"

    # Configure device
    if args.device == "cuda":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Check for predictions folder and create if it does not exist    
    if not os.path.exists("predictions/"):
        os.makedirs("predictions/", exist_ok=True)

    # TEMPORARY: load gold set
    gold = load_dataset("billsum", split="test").to_pandas()["summary"].tolist()

    # Generate dataset name from the test file path
    dataset_file = os.path.basename(args.testfile)
    dataset_dir = os.path.dirname(args.testfile)

    # Infer chunk index file
    data_index_file = os.path.join(dataset_dir, f"idx_{dataset_file}")

    # Infer max input length and max output length from data name since it is pre-chunked
    data_attr = dataset_file.split("_")
    chunk_strategy = data_attr[-1].split("-")
    # max_input_len, max_output_len = (int(chunk_strategy[1]), int(chunk_strategy[2]))
    max_input_len, max_output_len = (512, 512)

    # Configure model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
    model.config.num_beams = 2
    model.config.max_length = max_output_len
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    model_name = args.checkpoint.split("/")[1].split("-")[0]

    word_tok = AutoTokenizer.from_pretrained(args.checkpoint)

    # TODO: expand this to account for unchunked data 
    predictions_path = "predictions/" + f"{model_name}_{dataset_file}"

    test_dataset = pd.read_csv(args.testfile) # TODO: add dropna configuration
    test_index = pd.read_csv(data_index_file)

    # Check run mode
    mode = args.mode.lower()

    if mode == "e2e":
        pass
        # TODO: prepare to preprocess data before predicting
        # test_file_chunked_path = os.path.join(data_dir + model_name + "_" + args.dataset + "_test_set_chunked_" +
        #                                     str(int(max_input_length/2)) + "_" + str(max_input_length) + "_" +
        #                                     args.prev_loss + "_" + str(args.is_extractive))
        # test_file_chunked_idx_path = os.path.join(data_dir + model_name + "_" + args.dataset +
        #                                         "_test_set_chunked_idx_" + str(int(max_input_length/2)) + "_" +
        #                                         str(max_input_length) + "_" + args.prev_loss + "_" +
        #                                         str(args.is_extractive))

    elif mode == "predict":
        if os.path.exists(predictions_path):
            print("\nRead predictions already saved...\n")
            sys.exit()
        else:
            # Convert the datasets in Huggingface's Datasets format.
            test_dataset = Dataset.from_pandas(test_dataset)
            # Run model predictions
            predict_func = create_prediction(max_input_len, max_output_len, word_tok, model)

            if args.concat == "pre": 
                pass # TODO: implement pre concatenation
            else:               
                result = test_dataset.map(
                    predict_func,
                    batched=True,
                    batch_size=args.batch_size # TODO: make this configurable with config
                )

                # Reconstruct original lengths
                predictions = reconstruct(result["prediction"], test_index["idx"].tolist())

            # Evaluate metrics
            pred_scores = [eval_all(g, p) for g, p in zip(gold, predictions)]
            final = pd.DataFrame(pred_scores)

            # Add predictions back on
            final["predicted_summary"] = predictions

            # Write to CSV
            final.to_csv(predictions_path)            