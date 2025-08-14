import eval
from datasets import Dataset, load_dataset
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

# Set up module-level tqdm to write to STDERR like Dataset.map()
tqdm = lambda *args, **kwargs: _tqdm(*args, file=sys.stderr, **kwargs)

def reconstruct_by_doc_id(
        data_hf,
        k_limit: int=None,
        expect_confidence=False
):
    """
    Loop over a Dataset and reconstruct the final documents + generated summaries
    using the doc ID, ordered by the chunk index, as a Dataset
    """
    # Prepare defaultdict for grouped storage
    grouped_chunks = defaultdict(lambda: {
        "texts": [],
        "references": [],
        "predictions": [],
        "confidences": []
    })

    # Loop over dataset to group tuples of info by doc ID
    for row in data_hf:
        doc = grouped_chunks[row["doc_id"]]
        idx = row["chunk_idx"]
        # Append tuples (idx, content) to sort later
        doc["texts"].append((idx, row["text"]))
        doc["references"].append((idx, row["summary"]))

        # Get model confidence if it exists
        conf = row.get("model_confidence", None)
        doc["predictions"].append((idx, row["prediction"], conf))
    
    # Store final summaries
    final_data = []
    # Store problem outptus
    empty_results = []
    # Sort and reconstruct final documents
    for doc_id, chunks in grouped_chunks.items():
        # Always sort text and reference summary by index
        texts = [t for _, t in sorted(chunks["texts"], key=lambda x: x[0])]
        # Only include contentful strings for reconstruction with whitespace
        references = [r for _, r in sorted(chunks["references"], key=lambda x: x[0]) if len(r)]

        # Prepare generated predictions for reconstruction
        generated_predictions = chunks["predictions"]

        if k_limit is not None and expect_confidence:
            # Sort by confidence
            generated_predictions = sorted(generated_predictions, key=lambda x: x[2], reverse=True)
            # Apply top-k cutoff
            generated_predictions = generated_predictions[:k_limit]
        
        # Always sort by original index in-place
        # Only include contentful strings for reconstruction with whitespace
        predictions = [g for _, g, _ in sorted(generated_predictions, key=lambda x: x[0]) if len(g)]
        
        predicted_summary = " ".join(predictions)
        # Manually remove special tokens that are not [NO_SUMMARY]
        for special_token in ['<s>', '</s>', '<unk>', '<pad>', '<mask>']:
            predicted_summary = predicted_summary.replace(special_token, "")
        # Reconstruct with whitespace between
        row = {
                "doc_id": doc_id,
                "text": " ".join(texts),
                "summary": " ".join(references),
                "predicted_summary": predicted_summary
            }
        
        # Only add contentful predictions to final data for evaluation
        if len(predictions):
            final_data.append(row)
        else: 
            # Save empty results to a separate dataset 
            empty_results.append(row)

    # Return early if there is no final data -- hopefully unlikely
    if len(final_data) == 0:
        sys.exit("No summaries were generated. This setting does not work.")

    return Dataset.from_list(final_data), Dataset.from_list(empty_results)

def generate_blank_targets(
        data_skipped: Dataset,
        return_confidence=False
):
    """
    Receives skipped Dataset and generates blank target predictions.
    Optionally returns a model_confidence column (value 1.0) if requested
    """
    # Generate empty string predictions
    no_summary_preds = [""] * len(data_skipped)
    data_skipped = data_skipped.add_column("prediction", no_summary_preds)
    
    # Add confidence as needed
    if return_confidence:
        no_summary_conf = [1.0] * len(data_skipped)
        data_skipped = data_skipped.add_column("model_confidence", no_summary_conf)

    return data_skipped

def generate_predictions(
        model,
        tokenizer,
        max_output_length,
        data_hf,
        batch_size,
        device,
        return_confidence=False
):
    """
    Loop over a pre-tokenized Dataset to generate predictions, with scores if requested
    """
    # Store predictions and confidences as lists
    predictions = []
    confidences = []

    for start in tqdm(range(0, len(data_hf), batch_size), total=len(data_hf)//batch_size + 1):
        batch = data_hf[start:start+batch_size]

        # Prepare inputs batch with tokenized tensors on device
        inputs = {
            "input_ids": torch.tensor(batch["input_ids"]).to(device),
            "attention_mask": torch.tensor(batch["attention_mask"]).to(device),
        }

        # Add global attention mask if it exists
        if "global_attention_mask" in batch.keys():
            inputs.update({
                "global_attention_mask": torch.tensor(batch["global_attention_mask"]).to(device)
            })
        
        # Ensure no gradients are computed
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                output_scores=return_confidence,
                return_dict_in_generate=return_confidence,
                max_length=max_output_length,
                num_beams=2 # Hard-coded beam search
            )
        # Store confidences for each batch 
        if return_confidence:
            confidences.extend(outputs.sequences_scores.cpu().tolist())
            # Decode output tokens to text for each batch
            decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        else:
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Store decoded predictions
        predictions.extend(decoded)

    # Finally add columns to the Dataset
    data_hf = data_hf.add_column("prediction", predictions)
    if return_confidence:
        data_hf = data_hf.add_column("model_confidence", confidences)
    return data_hf

def compute_control_token_probability(
        model,
        data_hf,
        control_token_id,
        batch_size,
        device,
        p_limit: float=None
):
    """
    Loop over a pre-tokenized Dataset to compute full sequence probability of [NO_SUMMARY]
    and filter the data by p_limit into normal vs. skipped partitions if p_limit is provided
    """
    # Return early if no p limit is provided
    if p_limit is None:
        return None, data_hf
    
    # store cumulative probability of [NO_SUMMARY]
    control_ranks = []

    # Manually loop over data
    for start in tqdm(range(0, len(data_hf), batch_size), total=len(data_hf)//batch_size + 1):
        batch = data_hf[start:start+batch_size]

        # Prepare inputs batch with tokenized tensors on device
        inputs = {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long).to(device),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long).to(device),
        }

        # Add global attention mask if it exists
        if "global_attention_mask" in batch.keys():
            inputs.update({
                "global_attention_mask": torch.tensor(batch["global_attention_mask"], dtype=torch.long).to(device)
            })

        # Prepare decoder start token for step 1
        decoder_input = torch.full(
            (inputs["input_ids"].size(0), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        # Ensure gradient is not calculated
        with torch.no_grad():
            # Forward pass, get [batch_size x 1 x vocab_size]
            logits = model(
                **inputs, 
                decoder_input_ids=decoder_input
            ).logits[:, 0, :]

            # Compute relative rank of control token
            ranks = torch.argsort(logits, dim=-1, descending=True)
            control_positions = (ranks == control_token_id).nonzero(as_tuple=True)[1]
            relative_ranks = control_positions.float() / (logits.size(-1) - 1)

        control_ranks.extend(relative_ranks.cpu().tolist())
    # After loop finishes, add new column
    data_hf = data_hf.add_column("no_summary_rank", control_ranks)
    
    # Filter to split
    data_skipped = data_hf.filter(lambda ex: ex["no_summary_rank"] > p_limit)
    data_normal = data_hf.filter(lambda ex: ex["no_summary_rank"] <= p_limit)

    return data_skipped, data_normal

def tokenize_data(
        test_data: Dataset,
        updated_tokenizer: LEDTokenizer,
        batch_size: int,
        max_input_length: int,
        has_global_attn: bool = False
        ):
    """
    Map a Dataset of test examples with a text-tokenizing function, preserving 
    all original columns for downstream evaluation
    """
    
    tokenize_fn = utils.tokenize_text_factory(
        max_input_len=max_input_length,
        tokenizer=updated_tokenizer,
        has_global_attn=has_global_attn
        )
    test_data = test_data.map(
        tokenize_fn,
        batch_size=batch_size,
        batched=True
    )
    return test_data

def convert_data(test_data:pd.DataFrame):
    """Takes a pandas dataframe object, and converts it to a huggingface Dataset object.
    Modifies empty targets to be the custom token. Handles exclusion of the custom token.

    Arguments:
        test_data: The training data as a pandas.DataFrame
    Returns:
        the full test data as HF Dataset
    """
    # Log number of null rows to be dropped
    print(f"Detected {test_data.isna().sum().sum()} rows with null values, dropping...")
    # Drop any null rows
    test_data = test_data.dropna()
    # convert to HF Dataset type
    data_hf = Dataset.from_pandas(test_data)

    return data_hf

def update_model_tokenizer(
        model: LEDForConditionalGeneration,
        tokenizer: LEDTokenizer
    ):
    """Modifies the model vocabulary to include the custom 
    [NO_SUMMARY] token

    Arguments:
        model: Model object
        tokenizer: Associated tokenizer for model
    Returns:
        the model, tokenizer
    """
    print("Pretrained model special tokens")
    print(tokenizer.all_special_tokens)
    special_tokens_dict = {'additional_special_tokens': ["[NO_SUMMARY]"]}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "special tokens")

    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    model.resize_token_embeddings(len(tokenizer))

    print("Updated special tokens")
    print(tokenizer.all_special_tokens)
    return model, tokenizer, tokenizer.convert_tokens_to_ids("[NO_SUMMARY]")

def prepare_output_dirs(
        checkpoint_filepath: str,
        config_id: int
    ):
    """Prepares output directories for various finetuned models

    Arguments:
        checkpoint_filepath: TODO
        config_id: TODO
    Returns:
        None
    """
    # Log all paths to output file
    print(f"Finetuned model is loaded from {checkpoint_filepath}")

    # example: models/led-base/billsum_clean_train_se3-led-1024-512_simple/drop_blank_targets/1024_512_5_epochs/checkpoint-13922
    prediction_attrs = os.path.dirname(checkpoint_filepath).split("/")

    # set up output name
    prediction_filename = ".".join([str(config_id)] + prediction_attrs[-5:]) + ".csv"
    prediction_path = f"output/{prediction_filename}"
    empty_path = f"output/{str(config_id)}.EMPTY.csv"
    print(f"Test predictions will be saved to {prediction_path}")

    return prediction_path, empty_path

def load_model_tokenizer(
        checkpoint: str, 
        base_tokenizer: str = None
    ): 
    """Loads the model tokenizer and handles torch device assignment

    Arguments:
        checkpoint: name of finetuned checkpoint
        max_output_length: maximum number of output tokens
        base_model: name of base HuggingFace model for tokenizer loading
    Returns:
        model_name: simplified model name as a string for output directory
        model: LEDForConditionalGeneration object
        tokenizer: LEDTokenizer object
        device: torch.device object for saving Tensors
        has_global_attn: boolean for data preprocessing
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Argument for Longformer usage
    has_global_attn = "led-base" in checkpoint

    # Parse training input and output lengths from model checkpoint
    pattern = re.compile("se3-\w+-(\d+)-(\d+)")
    max_input_len = int(pattern.search(checkpoint).group(1))
    max_output_len = int(pattern.search(checkpoint).group(2))

    # Model name and settings
    model_name = '-'.join(checkpoint.split("/")[1].split("-")[0:2])
    model = LEDForConditionalGeneration.from_pretrained(checkpoint).to(device)
    model.config.num_beams = 2
    model.config.max_length = max_output_len
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    # Ensure model is in evaluation mode
    model.eval()

    # Instantiate tokenizer for base model if specified
    if base_tokenizer:
        tokenizer = LEDTokenizer.from_pretrained(base_tokenizer)
    else:
        tokenizer = LEDTokenizer.from_pretrained(checkpoint)

    return model_name, max_input_len, max_output_len, model, tokenizer, device, has_global_attn

def load_data(sourcefile: str):
    """Returns the training data as a pandas.Dataframe and the 
    max input and max output length as determined from the sourcefile name

    Arguments:
        sourcefile: file path of training data which includes info about 
        max input and max output length

    Returns:
        max input, max output, training data
    """
    # Parse input and output lengths
    pattern = re.compile("se3-\w+-(\d+)-(\d+)")
    max_input_len = int(pattern.search(sourcefile).group(1))
    max_output_len = int(pattern.search(sourcefile).group(2))
    
    # Load data and accompanying index file for preprocessing
    data = None
    data_idx = None
    idx_name = f"{os.path.dirname(sourcefile)}/idx_{os.path.basename(sourcefile)}"
    if os.path.isfile(sourcefile):
        print(f"Loading {sourcefile}...")
        data = pd.read_csv(sourcefile)
    else:
        print(f"Could not find {sourcefile}. Try again.")
        sys.exit(1)

    if os.path.isfile(idx_name):
        print(f"Loading {idx_name}...")
        data_idx = pd.read_csv(idx_name)
    else:
        print(f"Could not find {idx_name}. Try again.")
        sys.exit(1)
    
    # Merge index file on data to create relative index values
    # Rename columns
    data_idx = data_idx.rename(
        columns={
            "Unnamed: 0": "doc_id",
            "idx": "total_chunks"
        }
    )

    # Explode, reset index, merge onto data DF
    data_idx["chunk_idx"] = data_idx.total_chunks.apply(lambda x: list(range(x)))
    data_idx = data_idx.explode("chunk_idx")
    data_idx = data_idx.reset_index()
    data["doc_id"] = data_idx.doc_id
    data["chunk_idx"] = data_idx.chunk_idx
    data = data.rename(
        columns={
            "Unnamed: 0": "global_index"
        }
    )
    # Normalize empty reference summary rows with empty strings
    data.loc[data["summary"].isna(), "summary"] = ""

    return max_input_len, max_output_len, data

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=int, help="Specify inference/evaluation configuration ID (1, 2, 3)")
    parser.add_argument("--base_tokenizer", type=str, help="The base model tokenizer to reference")
    parser.add_argument("--checkpoint", default="google/pegasus-billsum", help="The local finetuned model checkpoint to evaluate")
    parser.add_argument("--mode", default="test", help="Specify dev or test mode")
    parser.add_argument("--testfile", type=str, help="The dataset to use for testing")
    parser.add_argument("--p_limit", type=float, default=None, help="Specify p-threshold for NO_SUMMARY control")
    parser.add_argument("--k_limit", type=int, default=None, help="Specify k value for final target selection")
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
    model_name, train_max_input_len, train_max_output_len, model, tokenizer, device, has_global_attn = load_model_tokenizer(
        checkpoint=args.checkpoint,
        base_tokenizer=args.base_tokenizer
    )
    # prepare output directories
    prediction_path, empty_path = prepare_output_dirs(
        checkpoint_filepath=args.checkpoint,
        config_id=args.config_id
    )
    ###########################################################################
    # Step 2: validate that test data and model have compatible I/O length
    ###########################################################################
    input_mismatch = train_max_input_len != test_max_input_len
    output_mismatch = train_max_output_len != test_max_output_len

    # Here we set input & output lengths after validating that test and training data were compatible
    if input_mismatch or output_mismatch:
        print("Train and test file do NOT have compatible input and/or output lengths. Try again.")
        sys.exit(1)
    else:
        max_input_len = train_max_input_len
        max_output_len = train_max_output_len
        print(f"Detected input length:{max_input_len} and output length:{max_output_len}")
    ###########################################################################
    # Step 3: Add special token to pretrained tokenizer
    ###########################################################################
    # update model + tokenizer vocab
    model, tokenizer, control_token_id = update_model_tokenizer(model, tokenizer)
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
        has_global_attn=has_global_attn
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
        device=device,
        return_confidence=return_confidence_scores
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
    # Step 9: Reconstruct full summaries
    ###########################################################################
    print("Reconstructing full summaries from generated predictions...")
    test_hf, test_empty = reconstruct_by_doc_id(
        data_hf=test_hf, 
        k_limit=args.k_limit, 
        expect_confidence=return_confidence_scores
    )
    
    # If there are empty rows, write them to a separate file
    if len(test_empty):
        test_empty.to_csv(empty_path)
    ###########################################################################
    # Step 10: Compute metrics and save output
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