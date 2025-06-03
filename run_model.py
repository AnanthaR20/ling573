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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
from tqdm.auto import tqdm
from utils import *

def warm_up(model, tokenizer):
    with torch.no_grad():
        for step in range(15):  # 15 steps to ensure stabilization
            inputs = get_led_warmup_input(tokenizer)
            _ = model.generate(**inputs, max_new_tokens=20)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="google/pegasus-billsum", help="The model checkpoint to use")
    parser.add_argument("--testfile", default="", help="The test filepath to use")
    parser.add_argument("--source", default="hub", help="Specify where model is sourced from")
    parser.add_argument("--concat", default="pre", help="Specify when to concatenate chunks")
    parser.add_argument("--mode", default="predict", help="To specify prediction or pipeline mode")
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size for generating predictions")
    parser.add_argument("--device", default="cuda", help="The device to use") 
    parser.add_argument("--model_name", default=None, help="Override checkpoint name if loading model from disk")
    # parser.add_argument("--max_input_len", type=int, default=2048, help="The input max size")
    # parser.add_argument("--max_output_len", type=int, default=512, help="The output max size")    
    args = parser.parse_args()

    # Set random seed
    set_seed(1234)
    # set_seed(args.seed)
    has_global_attn = args.checkpoint == "allenai/led-base-16384"

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
    # Split from file extension
    chunk_strategy = data_attr[-1].split(".")[0]
    if "simple" in chunk_strategy: # True for simple or simple_toy attributes
        # Change index from attributes
        chunk_strategy = data_attr[-2]
    # Chunking strategy is hyphen separated
    chunk_strategy = chunk_strategy.split("-")
    max_input_len, max_output_len = (int(chunk_strategy[-2]), int(chunk_strategy[-1]))
    # max_input_len, max_output_len = (512, 512)

    # Configure model and tokenizer
    model_name = None
    if args.source == "hub":
        print("Downloading model from HuggingFace...")
        model_name = '-'.join(args.checkpoint.split("/")[1].split("-")[0:2])
    elif args.source == "disk":
        print("Loading model from disk...")
        model_name = args.model_name
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
    model.config.num_beams = 2
    model.config.max_length = max_output_len
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    word_tok = AutoTokenizer.from_pretrained(args.checkpoint)

    # Warm up GPU
    warm_up(model, word_tok)

    # TODO: expand this to account for unchunked data 
    predictions_path = "predictions/" + f"{model_name}_{dataset_file}"

    test_dataset = pd.read_csv(args.testfile) 
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
            predict_func = create_prediction(max_input_len, max_output_len, word_tok, model, device, has_global_attn=has_global_attn)

            if args.concat == "pre": 
                pass # TODO: implement pre concatenation
            else:               
                result = test_dataset.map(
                    predict_func,
                    batched=True,
                    batch_size=args.batch_size,
                    # num_proc=8
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