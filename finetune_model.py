from datasets import load_dataset
from transformers import PegasusTokenizer
from transformers import DataCollatorForSeq2Seq
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import evaluate

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=256, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# GLOBALS
## DATA
billsum = load_dataset("billsum")

## MODEL
checkpoint = "google/pegasus-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
prefix = ""
tokenized_billsum = billsum["train"].map(preprocess_function, batched=True)
# tokenized_test_billsum = billsum["test"].map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

## EVAL
rouge = evaluate.load("rouge")

## FINE-TUNING
training_args = Seq2SeqTrainingArguments(
    output_dir="my_control_billsum_model",
    evaluation_strategy="no",
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_total_limit=3,
    max_steps=100000,
    # num_train_epochs=4,
    predict_with_generate=True,
    fp16=True, #change to bf16=True for XPU,
    label_smoothing_factor=0.1
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum,
    eval_dataset=billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
