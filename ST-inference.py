import torch
import csv
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate  # For BLEU and METEOR
import os
import torch
from functools import partial
from typing import List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# Load evaluation metrics
sacrebleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")

# Select source language and translation task
source_language = "tamil"  # Tamil
task = "translate"

# Select the language pair
language_pair = "ta_en"  # Tamil → English

# Load CoVoST 2 dataset
dataset = load_dataset("fixie-ai/covost2", language_pair, split="test")

# Load Whisper model for translation
model_name = "openai/whisper-small" 
# model_name = "kavyamanohar/whisper-ta-en-translation"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Get language token ID for Tamil
forced_decoder_ids = processor.get_decoder_prompt_ids(language=source_language, task=task)

# Prepare CSV file
output_file = f"translated_speech_{language_pair}.csv"
all_predictions = []
all_references = []

def process_single_sample(sample, processor, model, device, forced_decoder_ids) -> Tuple[str, str, str, float, float]:
    """Process a single sample and return its evaluation metrics."""
    audio = sample["audio"]["array"]
    ref_translation = sample["translation"]
    ta_text = sample["sentence"]
    
    # Process audio
    input_features = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generate translation
    with torch.no_grad():
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    
    whisper_translation = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    # Compute metrics
    bleu_score = sacrebleu.compute(
        predictions=[whisper_translation], 
        references=[[ref_translation]]
    )["score"]
    
    meteor_score = meteor.compute(
        predictions=[whisper_translation], 
        references=[ref_translation]
    )["meteor"] * 100
    
    return (
        ta_text,
        ref_translation,
        whisper_translation,
        round(bleu_score, 2),
        round(meteor_score, 2)
    )

def compute_overall_metrics(results: List[Tuple]) -> Tuple[float, float]:
    """Compute overall BLEU and METEOR scores from all results."""
    predictions = [result[2] for result in results]  # whisper translations
    references = [[result[1]] for result in results]  # reference translations
    
    overall_bleu = sacrebleu.compute(
        predictions=predictions, 
        references=references
    )["score"]
    
    overall_meteor = meteor.compute(
        predictions=predictions, 
        references=[ref[0] for ref in references]
    )["meteor"] * 100
    
    return overall_bleu, overall_meteor


with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Tamil Text",
        "Reference Translation",
        "Whisper Translation",
        "SacreBLEU",
        "METEOR"
    ])
    
    # Create partial function with fixed arguments
    process_fn = partial(
        process_single_sample,
        processor=processor,
        model=model,
        device=device,
        forced_decoder_ids=forced_decoder_ids
    )
    
    # Process samples using map
    results = list(map(
        process_fn,
        dataset
    ))
    
    # Write results to CSV
    writer.writerows(results)
    
    # Compute and print overall metrics
    overall_bleu, overall_meteor = compute_overall_metrics(results)
    print(f"Overall BLEU: {overall_bleu:.2f}")
    print(f"Overall METEOR: {overall_meteor:.2f}")

