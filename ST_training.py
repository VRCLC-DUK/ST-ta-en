from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate
import huggingface_hub
import torch
import os
import numpy as np

# Set NCCL debug environment variable
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Optional: Set memory growth
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Set memory growth to avoid memory fragmentation
    for device in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(0.9, device)

# Log in to Hugging Face Hub
huggingface_hub.login("TOKEN")

# Load dataset
language_pair = "ta_en"
dataset = load_dataset("fixie-ai/covost2", language_pair)

# Load Whisper model & processor
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(
    model_name,
    # Add gradient checkpointing for memory efficiency
    use_cache=False
)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define source language and task
source_language = "tamil"
task = "translate"

# Update model config for translation
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=source_language, task=task)
model.config.suppress_tokens = None

def preprocess_function(batch):
    # Process input audio
    audio = batch["audio"]["array"]
    
    # Get feature extractor output for audio
    input_features = processor.feature_extractor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.squeeze(0)
    
    # Tokenize target translation
    labels = processor.tokenizer(
        text=batch["translation"],
        padding="max_length",
        max_length=448,
        truncation=True,
        return_tensors="pt"
    ).input_ids.squeeze(0)
    
    return {
        "input_features": input_features,
        "labels": labels,
        "input_ids": labels
    }

# Apply preprocessing with smaller batch size
tokenized_dataset = dataset.map(
    preprocess_function,
    remove_columns=dataset["train"].column_names,
    num_proc=2,  # Reduced from 4 to 2
    batch_size=8  # Added explicit batch size
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.feature_extractor,
    model=model,
    padding=True
)
# Initialize BLEU metric
bleu_metric = evaluate.load("bleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Decode predicted tokens to text
    pred_str = processor.batch_decode(preds, skip_special_tokens=True)
    
    # Decode reference tokens to text
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)
    
    # Compute BLEU score
    # Need to wrap each reference in a list since BLEU expects multiple references
    label_str = [[ref] for ref in label_str]
    
    bleu_results = bleu_metric.compute(
        predictions=pred_str,
        references=label_str
    )
    
    return {
        "bleu_score": bleu_results["bleu"]
    }

# Update training arguments to include number of epochs and evaluation
training_args = Seq2SeqTrainingArguments(
    output_dir="./kavyamanohar/whisper-ta-en-translation",
    eval_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    save_steps=250,
    eval_steps=250,
    logging_steps=250,
    save_total_limit=2,
    learning_rate=1e-5,
    weight_decay=0.01,
    max_steps=5000,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
    hub_model_id="kavyamanohar/whisper-ta-en-translation",
    report_to="tensorboard",
    # Memory optimizations
    gradient_checkpointing=True,
    optim="adamw_torch",
    dataloader_num_workers=2,
    dataloader_pin_memory=False,
    # Add metric computation
    # metric_for_best_model="bleu_score",
    # greater_is_better=True,  # For BLEU score, higher is better
    load_best_model_at_end=True,
)

# Initialize trainer with compute_metrics
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=processor,
    compute_metrics=compute_metrics,  # Add compute_metrics function
)

# Train model
trainer.train()
processor.save_pretrained(training_args.output_dir)

# Push final model to Hugging Face Hub
trainer.push_to_hub()
processor.push_to_hub(training_args.hub_model_id)