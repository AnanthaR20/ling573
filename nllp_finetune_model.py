import argparse
import os
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
import utils
import re
import sys

def finetune(
        model: AutoModelForSeq2SeqLM, 
        model_path: str,
        model_name: str, 
        batch_size: int, 
        grad_acc: int, 
        num_epochs: int, 
        random_seed: int,
        train_data: Dataset,
        dev_data: Dataset
    ):
    """Starts with a model checkpoint and finetunes it on the training data.
    The resulting model is saved in the specified output directory.

    Arguments:
        model: Model object
        model_path: The output directory to save the finetuned model to
        model_name: The name of the model
        batch_size: size of training data batches
        grad_acc: adaptive gradient hyperparameter
        num_epochs: number of training rounds/batches to finetune on
        random_seed: random seed for replicability
        train_data: the training data as a HuggingFace Dataset object
        dev_data: development data partition (HuggingFace Dataset) for selecting the best model during training
    Returns:
        None
    """
    # Instantiate training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_path,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=num_epochs,
        logging_strategy="epoch",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        seed=random_seed,
        run_name=model_name
    )

    # Instantiate trainer, assuming pre-tokenized data
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
    )

    # Finetune!
    trainer.train()

    # Log dev loss
    eval_results = trainer.evaluate()
    print(f"Final evaluation: {eval_results}")
    return

def convert_data(train_data:pd.DataFrame, blank_target_setting:str):
    """Takes a pandas dataframe object, and converts it to a huggingface Dataset object.
    Modifies empty targets to be the custom token. Handles exclusion of the custom token.

    Arguments:
        train_data: The training data as a pandas.DataFrame
        blank_target_setting: "keep" or "drop" for inclusion/exclusion of the custom token [NO_SUMMARY],
                              "binary" to add two custom tokens 
    Returns:
        the full training data as HF Dataset
    """
    # Fill null summary examples according to blank_target_setting
    if blank_target_setting != "drop":
        # Filter the training data by any null document chunks (if any) as a precaution
        train_data = train_data.loc[train_data.text.notna()]
        
        # Add special positive token for non-null targets if using
        if blank_target_setting == "binary":
            train_data.loc[train_data.summary.notna(), 'summary'] = "[SUMMARIZE]" + train_data.loc[train_data.summary.notna(), 'summary']
        
        # Add the special negative token for null summary targets
        train_data.loc[train_data.summary.isna(), 'summary'] = "[NO_SUMMARY]"

    else:
        train_data = train_data.dropna() # Drop any row with ANY null values

    # convert to HF Dataset type
    data_hf = Dataset.from_pandas(train_data)

    return data_hf

def tokenize_split_data(
        data_hf: Dataset,
        updated_tokenizer: AutoTokenizer,
        batch_size: int,
        random_seed: int,
        max_input_length: int,
        max_output_length: int,
        has_global_attn: bool,
    ):
    """Tokenizes the data and splits data into train and dev sets

    Arguments:
        train_data: The training data as a HF Dataset
        updated_tokenizer: AutoTokenizer object (with special token added) for final data processing step
        batch_size: size of mapping batches
        random_seed: Random seed value to ensure consistent train/dev partitioning
        max_input_length: max input length
        max_output_length: max output length
        has_global_attn: boolean switch for global attention
    Returns:
        the training data and development data as HF Dataset
    """

    # tokenize data
    prepare_examples = utils.create_examples(
        max_input_len=max_input_length, 
        max_output_len=max_output_length,
        tokenizer=updated_tokenizer
    )
    
    # remove the original 'text' and 'summary' columns after mapping them to tokens
    examples = data_hf.map(
        prepare_examples,
        batch_size=batch_size,
        remove_columns=["text","summary"],
        batched=True
    )

    # Handles formatting for PyTorch
    examples.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"] if has_global_attn else
        ["input_ids", "attention_mask", "labels"]
    )
    # use train_test_split function to create a development partition
    data_dict = examples.train_test_split(
        train_size = 0.98,
        test_size = 0.02,
        seed= random_seed
    )
    data_dict["dev"] = data_dict["test"]
    del data_dict["test"]
    return data_dict["train"], data_dict["dev"]

def update_model_tokenizer(
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        blank_target_setting: str
    ):
    """Modifies the model vocabulary to include the custom 
    [NO_SUMMARY] token

    Arguments:
        model: Model object
        tokenizer: Associated tokenizer for model
        blank_target_setting: "keep" or "drop" for inclusion/exclusion in training data

    Returns:
        the model, tokenizer
    """
    print("Pretrained model special tokens")
    print(tokenizer.all_special_tokens)

    # Add only one token to tokenizer vocab unless using binary token setting
    if blank_target_setting != "binary":
        special_tokens_dict = {'additional_special_tokens': ["[NO_SUMMARY]"]}
    else:
        special_tokens_dict = {'additional_special_tokens': ["[NO_SUMMARY]", "[SUMMARIZE]"]}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "special tokens")

    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    model.resize_token_embeddings(len(tokenizer))

    print("Updated special tokens")
    print(tokenizer.all_special_tokens)
    return model, tokenizer

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
    return max_input_len, max_output_len, data

def prepare_output_dirs(
        model_name: str, 
        trainfile: str, 
        blank_target_setting: str,
        num_epochs: int, 
        max_input_length: int, 
        max_output_length:int 
    ):
    """Prepares output directories for various finetuned models

    Arguments:
        model_name: simple model name for output directory creation
        trainfile: full filepath for train file
        num_epochs: number of training epochs
        max_input_length: maximum number of input tokens
        max_output_length: maximum number of output tokens
    Returns:
        None
    """
    # example: billsum_clean_train_se3-led-2048-512_simple
    train_name = os.path.basename(trainfile).split(".")[0]
    model_dir = f"models/{model_name}/{train_name}/{blank_target_setting}_blank_targets"
    model_path = f"{model_dir}/{str(max_input_length)}_{str(max_output_length)}_{str(num_epochs)}_epochs"
    
    # Log all paths to output file
    print(f"Finetuned model will be saved to {model_path}")

    return model_path

def load_model_tokenizer(
        checkpoint: str, 
        max_output_length: int
    ): # could this function include the device assignment ~line 251
    """Loads the model tokenizer and handles torch device assignment

    Arguments:
        checkpoint: name of HuggingFace model
        max_output_length: maximum number of output tokens
    Returns:
        model_name: simplified model name as a string for output directory
        model: AutoModelForSeq2SeqLM object
        tokenizer: AutoTokenizer object
        device: torch.device object for saving Tensors
        has_global_attn: boolean for data preprocessing
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Argument for Longformer usage
    has_global_attn = "allenai/led" in checkpoint

    # Model name and settings
    model_name = '-'.join(checkpoint.split("/")[1].split("-")[0:2])
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    model.config.num_beams = 2
    model.config.max_length = max_output_length
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model_name, model, tokenizer, device, has_global_attn

def load_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint", default="google/pegasus-billsum", help="The model checkpoint to use")
    parser.add_argument("--mode", default="train", help="Specify training or pipeline mode")
    parser.add_argument("--trainfile", type=str, help="The dataset to use for training")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for training")
    parser.add_argument("--blank_targets", type=str, help="Specify how to use blank targets (drop or keep for control token usage)")
    parser.add_argument("--epochs", type=int, default=5, help="The number of epochs")
    parser.add_argument("--grad_acc", type=int, default=1, help="The gradient accumulation")
    parser.add_argument("--seed", type=int, default=1234, help="The seed to use")
    args = parser.parse_args()

    return args

def main():
    args = load_args()

    ############ INFERRED ARGUMENTS ###########
    # Set random seed
    set_seed(args.seed)

    ############################################

    # load TRAINFILE and TESTFILE, set max input/output length values
    # initialize in global scope
    
    # Parse input lengths from train data
    train_max_input_len, train_max_output_len, train_data = load_data(
        sourcefile=args.trainfile
    )

    max_input_len = train_max_input_len
    max_output_len = train_max_output_len
    print(f"Detected input length:{max_input_len} and output length:{max_output_len}")

    # load model, tokenizer 
    model_name, model, tokenizer, device, has_global_attn = load_model_tokenizer(
        checkpoint=args.checkpoint,
        max_output_length=max_output_len
    )

    # prepare output directories
    model_path = prepare_output_dirs(
        model_name=model_name,
        trainfile=args.trainfile,
        blank_target_setting=args.blank_targets,
        num_epochs=args.epochs,
        max_input_length=max_input_len,
        max_output_length=max_output_len
    )

    # update model + tokenizer vocab
    model, tokenizer = update_model_tokenizer(model, tokenizer, args.blank_targets)
    
    print("Adding special tokens before converting to Dataset...")
    # update training data with blank-target setting
    # convert to HF Dataset
    train_hf = convert_data(
        train_data=train_data,
        blank_target_setting=args.blank_targets
    )

    print("Tokenizing data...")
    # Tokenize and split the original train data into new train and dev sets
    train_hf, dev_hf = tokenize_split_data(
        data_hf=train_hf,
        updated_tokenizer=tokenizer,
        batch_size=args.batch_size,
        random_seed=args.seed,
        max_input_length=max_input_len,
        max_output_length=max_output_len,
        has_global_attn=has_global_attn
    )

    print("Finetuning...")
    # train model
    finetune(
        model=model,
        model_path=model_path,
        model_name=model_name,
        batch_size=args.batch_size,
        grad_acc=args.grad_acc,
        num_epochs=args.epochs,
        random_seed=args.seed,
        train_data=train_hf,
        dev_data=dev_hf
    )
    
    return

if __name__ == "__main__":
    main()