from datasets import Dataset, concatenate_datasets
from transformers import set_seed
import pandas as pd
import argparse
import re
import sys
import os
from eval import eval_metrics2 as metrics
from nllp_evaluate_model import *
from glob import glob

def select_k_chunks(test_empty, k, ascending_sort=None):
    """
    Args:
        test_empty: HF Dataset of input chunks that all predicted blank targets on first-pass inference
        k: number of chunks to keep in consideration for second-pass inference
    Returns:
        test_top_k: k * doc_ids rows of the test set for second-pass inference
        test_empty: remaining rows that will remain empty
    """
    temp = test_empty.to_pandas()

    top_k_rows = []
    empty_rows = []

    for _, group in temp.groupby("doc_id"):
        # If there are k or fewer chunks for this document, accept all of them
        if len(group) <= k:
            top_k_rows.extend(group.to_dict('records'))
        else:
            if ascending_sort is not None:
                # Sort the group by no_summary_rank
                ranked = group.sort_values('no_summary_rank', ascending=ascending_sort).to_dict('records')
                # Add the first k to top_k_rows
                top_k_rows.extend(ranked[0:k])

                # Add the remaining to empty_rows
                empty_rows.extend(ranked[k:])
            else:
                # Sample random k for second-pass inference
                random = group.sample(n=k)
                # Prepare the remaining rows that remain empty
                random_empty = group.loc[~group.index.isin(random.index)]
                # Add these rows to the final lists
                top_k_rows.extend(random.to_dict('records'))
                empty_rows.extend(random_empty.to_dict('records'))
    return Dataset.from_list(top_k_rows), Dataset.from_list(empty_rows)

def prepare_output_dir(checkpoint: str, new_config_id: int):
    """
    Args:
        checkpoint: model checkpoint path with attributes
        config_id: source config ID for predictions for postprocessing
    Returns:
        predictions path for 'new' config (original config ID + 2)
    """
    prediction_attrs = os.path.dirname(checkpoint).split("/")

    # set up output name
    prediction_filename = ".".join([str(new_config_id)] + prediction_attrs[-5:]) + ".csv"
    prediction_path = f"output/{prediction_filename}"
    print(f"Test predictions will be saved to {prediction_path}")

    return prediction_path

def load_chunk_level_predictions(config_id: int):
    """
    Args:
        config_id: corresponding to a one of our finetuning configurations
    Returns:
        test_hf: HF Dataset of chunk-level predictions
        test_empty: HF Dataset of chunks that all received blank predictions on first pass
        metadata: related metadata for these test data
    """
    metadata = {
        "max_input_length": None,
        "max_output_length": None,
        "se3_config": None
    }

    first_pass_chunks = None
    first_pass_eval = None
    test_empty = None
    pattern = re.compile("se3-\w+-(\d+)-(\d+)")

    print("Loading first-pass inference...")
    # Load chunk-level predictions
    for file in glob(f"output/chunked.{str(config_id)}.*.csv"):
        # Parse input and output lengths
        # ex: chunked.5.led-base.billsum_clean_train_se3-led-2048-512.binary_blank_targets.2048_512_5_epochs.checkpoint-45220.csv
        # se3_config: se3-led-2048-512
        metadata["se3_config"] = pattern.search(file).group(0)
        # max_input_length: 2048
        metadata["max_input_length"] = int(pattern.search(file).group(1))
        # max_output_length: 512
        metadata["max_output_length"] = int(pattern.search(file).group(2))
        data = pd.read_csv(file)
        # Normalize empty reference summary rows with empty strings
        data.loc[data["summary"].isna(), "summary"] = ""
        first_pass_chunks = Dataset.from_pandas(data)

    for file in glob(f"output/{str(config_id)}.*.csv"):
        data = pd.read_csv(file)
        first_pass_eval = Dataset.from_pandas(data)
    print("Loading source chunk-level inputs...")
    # Load input data
    _, _, source_data = load_data(f"preprocess/nllp_data/billsum_clean_test_{metadata['se3_config']}.csv")

    print("Loading document IDs for second-pass inference...")
    # Load document IDs with empty summaries
    for file in glob(f"output/{str(config_id)}.EMPTY.csv"):
        empty_df = pd.read_csv(file)
        # Extract empty_df["doc_id"].tolist()
        second_pass_doc_ids = empty_df["doc_id"].tolist()

        # Get all source_data rows that match this doc ID
        second_pass_mask = source_data["doc_id"].isin(second_pass_doc_ids)
        test_empty = Dataset.from_pandas(source_data[second_pass_mask])

    # Check if any are invalid before returning
    assert first_pass_chunks is not None
    assert first_pass_eval is not None
    assert test_empty is not None
    assert all([v is not None for v in metadata.values()])
    return first_pass_chunks, first_pass_eval, test_empty, metadata

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=int, help="Specify inference/evaluation configuration ID (1, 2, 3)")
    parser.add_argument("--new_config_id", type=int, help="Specify NEW inference/evaluation configuration ID (1, 2, 3)")
    parser.add_argument("--base_tokenizer", type=str, help="The base model tokenizer to reference")
    parser.add_argument("--checkpoint", default="google/pegasus-billsum", help="The local finetuned model checkpoint to evaluate")
    parser.add_argument("--k_selector", type=str, default=None, help="Specify control token for relative likelihood computation")
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
    ############################################

    ###########################################################################
    # Step 1: load data, model and inferred properties (input/output length)
    ###########################################################################
    first_pass_chunks, first_pass_eval, test_empty, test_metadata = load_chunk_level_predictions(args.config_id)

    # load model, validate train and test length values
    # load model, tokenizer 
    model, tokenizer, device, metadata = load_model_tokenizer(
        checkpoint=args.checkpoint,
        base_tokenizer=args.base_tokenizer
    )
    print(f"Running inference and evaluation for model:\n{metadata}")
    # prepare output directories
    prediction_path = prepare_output_dir(
        checkpoint=args.checkpoint,
        new_config_id=args.new_config_id
    )
    ###########################################################################
    # Step 2: validate that test data and model have compatible I/O length
    ###########################################################################
    input_mismatch = metadata["max_input_length"] != test_metadata["max_input_length"]
    output_mismatch = metadata["max_output_length"] != test_metadata["max_output_length"]

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
    print("Tokenizing data...")
    # tokenize input text column, keep other columns as-is
    test_empty = tokenize_data(
        test_data=test_empty, 
        updated_tokenizer=tokenizer, 
        batch_size=args.batch_size,
        max_input_length=max_input_len,
        has_global_attn=metadata["has_global_attn"]
        )
    
    ###########################################################################
    # Step 5: Select k chunks for each document set: 
    ###########################################################################
    # Re-compute relative likelihood because they were lost lol...
    if args.k_selector is not None:
        print(f"Re-computing relative probability rank for {args.k_selector} control token...")
        _, test_empty = compute_control_token_likelihood(
            model=model,
            data_hf=test_empty,
            control_token_id=tokenizer.convert_tokens_to_ids(args.k_selector),
            batch_size=args.batch_size,
            device=device
        )
    
    # Now select k chunks from each document group using the k_selector
    if args.k_selector == "[NO_SUMMARY]":
        print(f"Selecting min-{args.k_limit} chunks for each document set...")
        test_k, test_empty = select_k_chunks(test_empty, args.k_limit, False)
    elif args.k_selector == "[SUMMARIZE]":
        print(f"Selecting max-{args.k_limit} chunks for each document set...")
        test_k, test_empty = select_k_chunks(test_empty, args.k_limit, True)
    else:
        print(f"Selecting random-{args.k_limit} chunks for each document set...")
        test_k, test_empty = select_k_chunks(test_empty, args.k_limit)

    ###########################################################################
    # Step 6: generate predictions for top-k chunks
    ###########################################################################
    # Generate predictions as normal for rows where p_limit was not met
    
    print("Generating predictions while suppressing [NO_SUMMARIZE]...")
    test_k = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        max_output_length=max_output_len,
        data_hf=test_k,
        batch_size=args.batch_size,
        device=device,
        tokens_to_suppress=[control_token_id]
    )
    ###########################################################################
    # Step 7: Prune columns after generating predictions
    ###########################################################################
    # Remove LED token tensors because we don't need them anymore!
    unwanted_columns = ["input_ids", "attention_mask", "global_attention_mask"]
    existing_columns = set(test_k.column_names)
    columns_to_remove = [col for col in unwanted_columns if col in existing_columns]
    
    print(f"Removing columns {str(columns_to_remove)} that are not needed downstream...")
    test_k = test_k.remove_columns(columns_to_remove)

    ###########################################################################
    # Step 8: Recombine second-pass inference with remaining targets
    ###########################################################################   
    # Generate blank targets and tack them onto planned rows
    print("Generating blank targets for remaining chunks...")
    test_empty = test_empty.add_column("prediction", [""] * len(test_empty))
    
    # Concatenate test_empty and test_k
    print("Concatenating second-pass predictions to remaining blank targets...")
    second_pass = concatenate_datasets([test_k, test_empty])

    ###########################################################################
    # Step 9: Chunk-level classification metrics
    ###########################################################################
    print("Computing blank-target classification metrics across both first and second passes...")
    # Concatenate with first_pass (chunk level) to compute new IS metrics
    chunk_level_predictions = concatenate_datasets([first_pass_chunks, second_pass])
    metrics.get_decision_metrics(chunk_level_predictions["prediction"], chunk_level_predictions["summary"])

    ###########################################################################
    # Step 10: Reconstruct full summaries
    ###########################################################################
    print("Reconstructing full summaries from generated predictions...")
    second_pass, _ = reconstruct_by_doc_id(
        data_hf=second_pass, 
        special_tokens=tokenizer.all_special_tokens
    )
    
    ###########################################################################
    # Step 11: Compute metrics
    ###########################################################################
    print(f"Computing metrics for {len(second_pass)} generated summaries...")
    print("BERTScore...")
    second_pass = second_pass.map(
        lambda ex: metrics.get_bertscore_metrics(ex["predicted_summary"], ex["summary"]),
        batched=True,
        batch_size=args.batch_size
    )
    # Evaluate ROUGE, AlignScore, SummaC
    print("Starting AlignScore...")
    second_pass = second_pass.map(
        metrics.eval_alignscore_batch,
        batched=True,
        batch_size=args.batch_size
    )
    print("Starting ROUGE...")
    second_pass = second_pass.map(
        metrics.eval_rouge_batch,
        batched=True,
        batch_size=args.batch_size
    )
    # Evaluate LFTK
    print("Starting LFTK...")
    second_pass = second_pass.map(
        lambda ex: metrics.eval_lftk(ex["predicted_summary"], suffix=".GEN"),
        batched=False
    )

    ###########################################################################
    # Step 12: Concatenate with first-pass inference and save
    ###########################################################################
    print("Concatenating first-pass inference with second-pass...")
    test_hf = concatenate_datasets([first_pass_eval, second_pass])

    print("Computing new overall redundancy scores...")
    _, _, _, _ = metrics.get_redundancy_scores(test_hf["predicted_summary"])

    print("Saving predictions...")
    test_hf.to_csv(prediction_path)
    print("Postprocessing complete!")
    return

if __name__ == "__main__":
    main()