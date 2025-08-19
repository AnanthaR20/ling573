import torch

################################################################################
# NLLP utils functions
################################################################################

def generate_prediction_factory(
        model,
        tokenizer,
        max_output_length,
        device,
        tokens_to_suppress,
        skip_special_tokens,
        num_beams=2
    ):
    def factory(examples):
        # Prepare inputs batch with tokenized tensors on device
        inputs = {
            "input_ids": torch.tensor(examples["input_ids"]).to(device),
            "attention_mask": torch.tensor(examples["attention_mask"]).to(device),
        }

        # Add global attention mask if it exists
        if "global_attention_mask" in examples.keys():
            inputs.update({
                "global_attention_mask": torch.tensor(examples["global_attention_mask"]).to(device)
            })
        
        # Prepare to suppress [NO_SUMMARY] if it was provided
        if len(tokens_to_suppress):
            inputs.update({
                "suppress_tokens": tokens_to_suppress
            })

        # Ensure no gradients are computed
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_output_length,
                num_beams=num_beams # Hard-coded beam search
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)
        return {
            "prediction": decoded
        }
    return factory

def tokenize_text_factory(
        max_input_len: int,
        tokenizer,
        has_global_attn: bool
):
    def factory(examples):
        # tokenize and return tensors so we can create a global attention mask
        inputs = tokenizer(examples["text"], padding="max_length", 
                        truncation=True, max_length=max_input_len,
                        return_tensors="pt")
        examples["input_ids"] = inputs.input_ids
        examples["attention_mask"] = inputs.attention_mask
        if has_global_attn:
            global_attention_mask = torch.zeros_like(examples["input_ids"])   
            global_attention_mask[:, 0] = 1
            examples["global_attention_mask"] = global_attention_mask
        return examples
    return factory
################################################################################
# As written in Spring 2025, used for NLLP
################################################################################
def create_examples(max_input_len, max_output_len, tokenizer):
    def process_data_to_model_inputs(examples):
        inputs = tokenizer(examples["text"], padding="max_length", max_length=max_input_len, truncation=True)
        outputs = tokenizer(examples["summary"], padding="max_length", max_length=max_output_len, truncation=True)
        examples["input_ids"] = inputs.input_ids
        examples["attention_mask"] = inputs.attention_mask
        examples["global_attention_mask"] = len(examples["input_ids"]) * [
            [0 for _ in range(len(examples["input_ids"][0]))]
        ]
        examples["global_attention_mask"][0][0] = 1
        examples["labels"] = outputs.input_ids
        examples["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in examples["labels"]
        ]
        return examples
    
    return process_data_to_model_inputs

################################################################################
# DEPRECATED FROM SPRING 2025
################################################################################
def reconstruct(preds, data_index):
    final_summaries = []
    rows_seen = 0

    for i in data_index: 
        chunk_count = int(i)
        summary = ""
        for j in range(chunk_count):
            summary += preds[rows_seen + j].strip()
        final_summaries.append(summary)
        rows_seen += chunk_count # update the number of rows seen
    return final_summaries

def create_prediction(max_input_len, max_output_len, tokenizer, model, device, has_global_attn=False):
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
        outputs = None

        if has_global_attn:
            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = 1
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                global_attention_mask=global_attention_mask,
                                                max_length=max_output_len, num_beams=2)
        else:
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                max_length=max_output_len, num_beams=2)

        return {
            "prediction": tokenizer.batch_decode(outputs, skip_special_tokens=True)
        }
    return predict