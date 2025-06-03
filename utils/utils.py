import torch

def warm_up(model, tokenizer):
    with torch.no_grad():
        for step in range(15):  # 15 steps to ensure stabilization
            inputs = get_led_warmup_input(tokenizer)
            _ = model.generate(**inputs, max_new_tokens=20)
    return

def get_led_warmup_input(tokenizer):
    text = "This is a warm-up sequence. " * 160  # ~2048 tokens
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to("cuda")
    inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
    inputs["global_attention_mask"][:, [0]] = 1  # Global attention on [CLS]
    return inputs

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

def create_simplify(model, tokenizer, max_input_len, max_output_len):
    def simplify(example):
        # Tokenize input
        tokens = tokenizer(
            example["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_input_len, 
            return_tensors='pt'
        )
        # Generate output
        output_ids = model.generate(
        tokens['input_ids'], 
        attention_mask=tokens['attention_mask'], 
        max_length=max_output_len, 
        num_beams=5
        )
        # Decode output
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # Overwrite text content with decoded content
        example["text"] = decoded
        return example
    return simplify

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

# TEMPORARY: move this function out of se3 to avoid Python path errors
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