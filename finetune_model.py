import argparse
import os
import shutil
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
from eval.eval_metrics import eval_all
from tqdm.auto import tqdm
from utils.utils import *


def load_data(source_name):
    data = None
    data_idx = None
    idx_name = f"{os.path.dirname(source_name)}/idx_{os.path.basename(source_name)}"
    if os.path.isfile(source_name):
        print(f"Loading {source_name}...")
        data = pd.read_csv(source_name)
    else:
        print(f"Could not find {source_name}. Try full pipeline mode.")

    if os.path.isfile(idx_name):
        print(f"Loading {idx_name}...")
        data_idx = pd.read_csv(idx_name)
    else:
        print(f"Could not find {idx_name}. Try full pipeline mode.")
    return data, data_idx

def test_predictions(max_input_length, max_output_length, tokenizer, model, device, ds, has_global_attn):
    predict_func = create_prediction(max_input_length, max_output_length, tokenizer, model, device, has_global_attn=has_global_attn)
    result = ds.map(
        predict_func,
        batched=True,
        batch_size=args.batch_size
    )
    return result

def prepare_data(process_func, ds, has_global_attn):
    examples = ds.map(
                process_func,
                batched=True,
                batch_size=args.batch_size,
                remove_columns=["text", "summary"]
            )
    examples.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "global_attention_mask", "labels"] if has_global_attn else
                ["input_ids", "attention_mask", "labels"]
            )
    return examples

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=None, help="Specify finetuning strategy")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for training")
    parser.add_argument("--mode", default="train", help="Specify training or pipeline mode")
    parser.add_argument("--checkpoint", default="google/pegasus-billsum", help="The model checkpoint to use")
    parser.add_argument("--trainfile", type=str, help="The dataset to use for training")
    parser.add_argument("--testfile", type=str, help="The dataset to use for test evaluation")
    parser.add_argument("--epochs", type=int, default=5, help="The number of epochs")
    parser.add_argument("--grad_acc", type=int, default=1, help="The gradient accumulation")
    parser.add_argument("--max_input_len", type=int, default=1024, help="The input max size")
    parser.add_argument("--max_output_len", type=int, default=512, help="The output max size")
    parser.add_argument("--seed", type=int, default=1234, help="The seed to use")
    parser.add_argument("--device", default="cuda", help="The device to use")
    parser.add_argument("--test", action="store_true", default=False, help="Specify if only testing")
    parser.add_argument("--concat", default="post", help="Specify concatenation strategy")
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Argument for Longformer usage
    has_global_attn = args.checkpoint == "allenai/led-base-16384"

    # Set device name
    if args.device == "cuda":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    print(f"Running on device: {device}")

    # Set max input and output lengths
    max_input_length = args.max_input_len
    max_output_length = args.max_output_len

    # Model name and settings
    model_name = '-'.join(args.checkpoint.split("/")[1].split("-")[0:2])
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
    model.config.num_beams = 2
    model.config.max_length = max_output_length
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    # Instantiate tokenizer
    word_tok = AutoTokenizer.from_pretrained(args.checkpoint)

    # Set up predictions and model filepaths
    if not os.path.exists("predictions/"):
        os.makedirs("predictions/", exist_ok=True)

    train_data = os.path.basename(args.trainfile).split(".")[0]
    test_data = os.path.basename(args.testfile).split(".")[0]
    predictions_path = f"predictions/{model_name}_{train_data}_{test_data}_{str(max_input_length)}_{str(max_output_length)}_{str(args.epochs)}_epochs"
    model_dir = f"models/{model_name}/{train_data}"
    model_path = f"{model_dir}/{str(max_input_length)}_{str(max_output_length)}_{str(args.epochs)}_epochs"
    
    # Full pipeline mode
    if args.mode == "pipeline": 
        pass # TODO: implement full pipeline
        # Load dataset, preprocess before training
    elif args.mode == "train": # Loads dataset, assumes it's in the correct format
        print(f"\nWe are using the model '{model_name}' for the dataset '{args.trainfile}' with input truncation\n")
        print(f"\nInput size: '{max_input_length}' - Output size '{max_output_length}'\n")
        print(f"Finetuned model will be saved to {model_path}\n")
        print(f"Finetuned model predictions will be saved to {predictions_path}\n")

        train_examples, train_idx = load_data(args.trainfile)
        test_examples, test_idx = load_data(args.testfile)

        print("Dropping irrelevant chunks from training data")
        train_examples = train_examples.dropna()
        print("Converting to Huggingface's Dataset format")
        train_hf = Dataset.from_pandas(train_examples)
        test_hf = Dataset.from_pandas(test_examples)

    # Check if predictions have already been generated
    if not os.path.isfile(predictions_path) or args.test:
        # Load trained weights if they exist
        if os.path.isfile(model_path) and not args.test:
            print("\nLoad trained weights\n")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else: # Otherwise, prepare to train the model
            print("\nPrepare the training and evaluation sets...\n")
            process_inputs = create_examples(max_input_length, max_output_length, word_tok)
            train_dataset_model_input = prepare_data(process_inputs, train_hf, has_global_attn)
            # eval_dataset_model_input = prepare_data(test_examples, has_global_attn, max_input_length, max_output_length)

            training_args = Seq2SeqTrainingArguments(
                output_dir=model_path,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_acc,
                num_train_epochs=args.epochs,
                logging_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                seed=args.seed,
                # fp16=True,
                run_name=model_name
            )

            trainer = Seq2SeqTrainer(
                model=model,
                tokenizer=word_tok,
                args=training_args,
                train_dataset=train_dataset_model_input,
                # eval_dataset=eval_dataset_model_input,
                # compute_metrics=compute_metrics,
                # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

            print("\nStart Training...\n")
            trainer.train()

        print("\nStart Evaluation...\n")
        # Run model predictions
        if args.concat == "pre": 
            # Reconstruct full documents using document chunk index
            test_source_simple = reconstruct(test_examples["text"], test_idx["idx"].tolist())
            # Cast these as Huggingface datasets TODO: check that this data structure is what we expect
            test_docs_simple = Dataset.from_list(test_source_simple)
            # Generate predictions
            predictions = test_predictions(max_input_length, max_output_length, word_tok, model, device, test_docs_simple)
        else:               
            result = test_predictions(max_input_length, max_output_length, word_tok, model, device, test_hf, has_global_attn)

            # Reconstruct original lengths
            predictions = reconstruct(result["prediction"], test_idx["idx"].tolist())

        # Evaluate metrics
        gold = load_dataset("billsum", split="test").to_pandas()["summary"].tolist()
        pred_scores = [eval_all(g, p) for g, p in zip(gold, predictions)]
        final = pd.DataFrame(pred_scores)

        # Add predictions back on
        final["predicted_summary"] = predictions

        # Write to CSV
        final.to_csv(predictions_path)

        if args.strategy != None:
            print(f"Compressing model to {args.strategy}.zip\n")
            shutil.make_archive(f"{args.strategy}", 'zip', root_dir=model_dir, base_dir=model_path)
            print("Removing full model files...\n")
            shutil.rmtree(model_path)            
    else:
        print("\nPredictions already saved, exiting...\n")
