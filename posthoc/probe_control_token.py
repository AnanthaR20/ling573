from transformers import set_seed
import sys
sys.path.insert(0, ".")
from nllp_evaluate_model import *

def main():
    args = load_args()

    ############ INFERRED ARGUMENTS ###########
    # Set random seed
    set_seed(args.seed)

    # generate controlled summaries with model confidence scores for k_limit if provided
    return_confidence_scores = args.k_limit is not None
    print(f"Inference will use model confidence? {str(return_confidence_scores).upper()}")
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
    print("Computing logits for [NO_SUMMARY] control token...")
    _, test_hf = compute_control_token_probability(
        model=model,
        data_hf=test_hf,
        control_token_id=control_token_id,
        batch_size=args.batch_size,
        device=device,
        p_limit=args.p_limit
    )
    ###########################################################################
    # Step 6: Prune columns
    ###########################################################################
    # Remove LED token tensors because we don't need them anymore!
    unwanted_columns = ["input_ids", "attention_mask", "global_attention_mask"]
    existing_columns = set(test_hf.column_names)
    columns_to_remove = [col for col in unwanted_columns if col in existing_columns]
    
    print(f"Removing columns {str(columns_to_remove)} that are not needed downstream...")
    test_hf = test_hf.remove_columns(columns_to_remove)
    # print(test_hf)
    ###########################################################################
    # Step 6: Analyze control token probability
    ###########################################################################
    df = test_hf.to_pandas()

    df.to_csv(f"{str(args.config_id)}_PROBE.csv")

    print("Statistics for [NO_SUMMARY] rank for true cases:")
    print(df.loc[~df.summary.str.len().astype(bool), "no_summary_rank"].describe())

    print("Statistics for [NO_SUMMARY] rank for false cases:")
    print(df.loc[df.summary.str.len().astype(bool), "no_summary_rank"].describe())


if __name__ == "__main__":
    main()