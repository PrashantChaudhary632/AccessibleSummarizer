# Accessible Legal AI Summarizer

A hierarchical transformer-based system for summarizing long Indian legal and government policy documents, with an emphasis on accessibility and readability.

## Features

- Abstractive summarization using Hugging Face transformer models
- BART-large-CNN as the default summarization model
- Optional T5-small model selection
- Text preprocessing and summary post-processing utilities
- ROUGE-based evaluation support
- Human evaluation forms, source excerpts, and reference summaries

## Project Structure

```text
AccessibleSummarizer/
├── summarizer.py
├── accessibility_layer.py
├── baseline_rouge.py
├── evaluate_rouge.py
├── preprocessing/
├── evaluation/
├── evaluation_scripts/
├── Publishing Documents/
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or a compatible Python version
- pip
- Sufficient RAM and storage for the selected transformer model
- GPU recommended for faster inference; CPU inference may be slower

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

The summarizer expects cleaned text files under `dataset/ready_text` by default and writes summaries and metrics under the `output` directory.

After preparing the input directory, run:

```bash
python summarizer.py
```

Select the T5-small model:

```bash
python summarizer.py --model t5
```

Run with the optional ROUGE evaluation:

```bash
python summarizer.py --rouge
```

For command-line options:

```bash
python summarizer.py --help
```

## Dataset

The dataset is not included in this repository. Prepare your own appropriately sourced legal and policy text files in the expected input directory before running the pipeline.

## Evaluation

The `evaluation/` directory contains human-evaluation materials, including reference summaries and source excerpts. The `evaluation_scripts/` directory contains scripts supporting the evaluation workflow.

## Research

This project is associated with the research paper:

**Towards Accessible Legal AI: A Hierarchical Transformer-Based Approach for Summarizing Long Indian Legal and Policy Documents**

See the `Publishing Documents/` directory for selected publication-related documents.

## Limitations

Automated ROUGE scores do not fully measure factual correctness, legal completeness, or accessibility. Generated summaries should be reviewed before being used for legal or policy decisions.

## License

Add an appropriate license before distributing or reusing this project. Ensure you have permission to publish any included documents, evaluation materials, and third-party content.
