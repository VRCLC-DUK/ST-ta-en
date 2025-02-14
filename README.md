# Tamil-English Speech Translation Project

This project implements an end-to-end speech translation system that directly translates Tamil speech to English text using the Whisper model. The system is designed for both training and inference purposes.

## Overview

The project consists of three main components:
- Training script (`ST-train.py`)
- Inference script (`ST-inference.py`)  
- Dependencies file (`requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd tamil-english-st
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

The project requires the following major packages:
- PyTorch: Deep learning framework
- Transformers: Hugging Face's transformers library for Whisper model
- Datasets: For data loading and processing
- SacreBLEU: For translation quality evaluation
- METEOR: For additional translation metrics
- NumPy: For numerical computations
- pandas: For data manipulation

For the complete list of dependencies with specific versions, refer to `requirements.txt`.

## Training (`ST-train.py`)

The training script implements fine-tuning of the Whisper model for Tamil-to-English speech translation.

### Features
- Fine-tunes Whisper model on Tamil speech data
- Includes validation during training
- Saves checkpoints and training logs
- Computes BLEU and METEOR scores

### Usage
```bash
python ST-train.py
```

## Inference (`ST-inference.py`)

The inference script handles translation of Tamil speech to English text using the trained model.

### Features
- Loads trained Whisper model from HF Hub
- Processes audio input
- Generates English translations
- Computes translation quality metrics
- Supports batch processing
- Saves results to CSV format

### Usage
```bash
python ST-inference.py 
```

## Output Format

The inference script generates a CSV file with the following columns:
- Tamil Text: Original Tamil text (if available)
- Reference Translation: Ground truth English translation
- Whisper Translation: Model-generated translation
- SacreBLEU: BLEU score for the translation
- METEOR: METEOR score for the translation

## Model Evaluation

The system evaluates translations using:
- SacreBLEU: Industry-standard BLEU score implementation
- METEOR: Additional metric for translation quality
- Individual and overall scores are computed for both metrics


## License

MIT


