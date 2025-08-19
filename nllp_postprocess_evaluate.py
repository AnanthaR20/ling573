from datasets import Dataset
from transformers import LEDForConditionalGeneration, LEDTokenizer
from transformers import set_seed
import torch
import pandas as pd
from tqdm.auto import tqdm as _tqdm
import argparse
import re
import sys
import os
from collections import defaultdict
import utils
from eval import eval_metrics2 as metrics
from nllp_evaluate_model import *

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=int, help="Specify inference/evaluation configuration ID (1, 2, 3)")
    parser.add_argument("--base_tokenizer", type=str, help="The base model tokenizer to reference")
    parser.add_argument("--checkpoint", default="google/pegasus-billsum", help="The local finetuned model checkpoint to evaluate")
    parser.add_argument("--p_limit", type=float, default=None, help="Specify p-threshold for NO_SUMMARY control")
    parser.add_argument("--k_limit", type=int, default=2, help="Specify k value for final target selection")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for training")
    parser.add_argument("--seed", type=int, default=1234, help="The seed to use")
    args = parser.parse_args()

    return args

def main():
    args = load_args()

    ############ INFERRED ARGUMENTS ###########
    # Set random seed
    set_seed(args.seed)

    # generate controlled summaries with model confidence scores for k_limit if provided
    return_confidence_scores = args.k_limit is not None
    print(f"Inference will use model confidence? {str(return_confidence_scores).upper()}")
    ############################################

    ###########################################################################
    # Step 1: load data, model and inferred properties (input/output length)
    ###########################################################################
    test_max_input_len, test_max_output_len, test_data = load_data(
        sourcefile=args.testfile
    )

    # load model, validate train and test length values
    # load model, tokenizer 
    model, tokenizer, device, metadata = load_model_tokenizer(
        checkpoint=args.checkpoint,
        base_tokenizer=args.base_tokenizer
    )
    print(f"Running inference and evaluation for model:\n{metadata}")
    # prepare output directories
    prediction_path, empty_path = prepare_output_dirs(
        checkpoint_filepath=args.checkpoint,
        config_id=args.config_id
    )
    ###########################################################################
    # Step 2: validate that test data and model have compatible I/O length
    ###########################################################################
    input_mismatch = metadata["max_input_length"] != test_max_input_len
    output_mismatch = metadata["max_output_length"] != test_max_output_len

    # Here we set input & output lengths after validating that test and training data were compatible
    if input_mismatch or output_mismatch:
        print("Train and test file do NOT have compatible input and/or output lengths. Try again.")
        sys.exit(1)
    else:
        max_input_len = metadata["max_input_length"]
        max_output_len = metadata["max_output_length"]
        print(f"Detected input length:{max_input_len} and output length:{max_output_len}")
    ###########################################################################
    # Step 3: Add special token to pretrained tokenizer
    ###########################################################################
    # update model + tokenizer vocab
    model, tokenizer, control_token_id = update_model_tokenizer(
        model, 
        tokenizer, 
        metadata["blank_target_setting"]
        )
    ###########################################################################
    # Step 4: Prepare Se3-ed test data for ingestion
    ###########################################################################
    print("Converting CSV to HuggingFace dataset...")
    # convert to HF Dataset - no manipulation of blank targets in test setting
    test_hf = convert_data(
        test_data=test_data
    )

    print("Tokenizing data...")
    # tokenize input text column, keep other columns as-is
    test_hf = tokenize_data(
        test_data=test_hf, 
        updated_tokenizer=tokenizer, 
        batch_size=args.batch_size,
        max_input_length=max_input_len,
        has_global_attn=metadata["has_global_attn"]
        )
    ###########################################################################
    # Step 5: use p-threshold as a filter
    ###########################################################################
    # if p_limit is provided, compute [NO_SUMMARY] probability and split data
    print("Computing relative probability rank for [NO_SUMMARY] control token...")
    test_skipped, test_hf = compute_control_token_probability(
        model=model,
        data_hf=test_hf,
        control_token_id=control_token_id,
        batch_size=args.batch_size,
        device=device,
        p_limit=args.p_limit
    )
    ###########################################################################
    # Step 6: generate predictions for "normal rows"
    ###########################################################################
    # Generate predictions as normal for rows where p_limit was not met
    print("Generating predictions...")
    test_hf = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        max_output_length=max_output_len,
        data_hf=test_hf,
        batch_size=args.batch_size,
        device=device
    )
    ###########################################################################
    # Step 7: generate blank targets for filtered rows
    ###########################################################################
    # If there are skippable rows, override prediction with blank targets
    if test_skipped is not None and len(test_skipped):
        print("Generating [NO_SUMMARY] targets...")
        test_skipped = generate_blank_targets(test_skipped, return_confidence_scores)
    
        # Minimal validation that we didn't mess up earlier
        if set(test_skipped.column_names) != set(test_hf.column_names):
            raise ValueError("Partitions with normal targets vs. blank targets do not have the same columns")
        
        # Then we concatenate datasets of the same shape
        print("Concatenating all targets...")
        test_hf = test_hf.concatenate(test_skipped)
    ###########################################################################
    # Step 8: Prune columns before moving on
    ###########################################################################
    # Remove LED token tensors because we don't need them anymore!
    unwanted_columns = ["input_ids", "attention_mask", "global_attention_mask"]
    existing_columns = set(test_hf.column_names)
    columns_to_remove = [col for col in unwanted_columns if col in existing_columns]
    
    print(f"Removing columns {str(columns_to_remove)} that are not needed downstream...")
    test_hf = test_hf.remove_columns(columns_to_remove)

    ###########################################################################
    # Step 9: Chunk-level classification metrics
    ###########################################################################
    print("Computing blank-target classification metrics...")
    metrics.get_decision_metrics(test_hf["prediction"], test_hf["summary"])

    # TEMPORARY
    print("Saving readable chunk-level results...")
    chunk_predictions_path = os.path.dirname(prediction_path) + "/chunked." + os.path.basename(prediction_path)
    test_hf.to_csv(chunk_predictions_path)

    ###########################################################################
    # Step 10: Reconstruct full summaries
    ###########################################################################
    print("Reconstructing full summaries from generated predictions...")
    test_hf, test_empty = reconstruct_by_doc_id(
        data_hf=test_hf, 
        k_limit=args.k_limit, 
        expect_confidence=return_confidence_scores,
        special_tokens=tokenizer.all_special_tokens
    )
    
    # If there are empty rows, write them to a separate file
    if len(test_empty):
        test_empty.to_csv(empty_path)
    ###########################################################################
    # Step 11: Compute metrics and save output
    ###########################################################################
    print(f"Computing metrics for {len(test_hf)} generated summaries...")
    print("BERTScore...")
    test_hf = test_hf.map(
        lambda ex: metrics.get_bertscore_metrics(ex["predicted_summary"], ex["summary"]),
        batched=True,
        batch_size=args.batch_size
    )
    # Evaluate ROUGE, AlignScore, SummaC
    print("Starting AlignScore...")
    test_hf = test_hf.map(
        metrics.eval_alignscore_batch,
        batched=True,
        batch_size=args.batch_size
    )
    print("Starting ROUGE...")
    test_hf = test_hf.map(
        metrics.eval_rouge_batch,
        batched=True,
        batch_size=args.batch_size
    )
    # # print("SummaC...")
    # # test_hf = test_hf.map(
    # #     metrics.eval_summac_batch,
    # #     batched=True,
    # #     batch_size=args.batch_size
    # # )

    # Evaluate LFTK
    print("Starting LFTK...")
    test_hf = test_hf.map(
        lambda ex: metrics.eval_lftk(ex["predicted_summary"], suffix=".GEN"),
        batched=False
    )
    print("Computing overall redundancy scores...")
    _, _, _, _ = metrics.get_redundancy_scores(test_hf["predicted_summary"])

    print("Saving predictions...")
    test_hf.to_csv(prediction_path)
    return

if __name__ == "__main__":
    main()