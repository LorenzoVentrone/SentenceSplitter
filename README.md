# Sentence Splitter (Token Classification)

This repository contains a multilingual sentence boundary detection pipeline built as a token classification task. The model predicts end-of-sentence boundaries at token level (`0` = not end of sentence, `1` = end of sentence).

## What Changed

The dataset pipeline now merges three sources into a single unified training corpus:

1. Professor dataset from `sent_split_data.tar.gz` (only `*-train.sent_split` files with `<EOS>` tags)
2. Legal JSONL datasets from `MultiLegalSBD` (only `*train.jsonl` files)
3. General-domain Wikipedia data (Italian and English)

This replaced the previous setup based only on two legal JSONL files.

## Data Processing Strategy

All data sources are normalized to the same output schema:

- `tokens`: list of token strings
- `ner_tags`: list of integer labels aligned with `tokens`

### 1) TAR Dataset (`.sent_split`)

`Dataset.py` calls `utils.process_tar_dataset(...)`.

- Each token containing `<EOS>` is cleaned and labeled `1`
- All other tokens are labeled `0`
- Only files ending with `-train.sent_split` are included
- Documents are chunked with sliding window:
   - `WINDOW_SIZE = 128`
   - `STRIDE = 100`

### 2) MultiLegalSBD JSONL Datasets

`Dataset.py` scans `MultiLegalSBD` with `*train.jsonl` and processes each file through `utils.process_jsonl_dataset(...)`.

Important handling:

- Only spans with `label == "Sentence"` are considered
- `token_end` is mapped to the last non-whitespace token
- Whitespace-only tokens are skipped to keep alignment consistent with TAR processing

### 3) Wikipedia (IT + EN)

`Dataset.py` calls `utils.process_wikipedia_dataset(...)` for:

- `20231101.it`
- `20231101.en`

How it is handled:

- Articles are downloaded from `wikimedia/wikipedia`
- Text is split into paragraphs
- Sentences are detected with NLTK (`sent_tokenize`)
- Tokens are built with whitespace split (to match the project tokenization style)
- Last token of each detected sentence is labeled `1`

## Final Unified Dataset

After merging all sources, the resulting dataset is converted to Hugging Face format and saved locally.

- Output folder: `unified_training_dataset`
- Features: `tokens`, `ner_tags`

Final split used for training/evaluation:

```python
DatasetDict({
      train: Dataset({
            features: ['tokens', 'ner_tags'],
            num_rows: 19229
      })
      test: Dataset({
            features: ['tokens', 'ner_tags'],
            num_rows: 2137
      })
})
```

## Current Training Configuration

Latest run configuration for this version:

```python
TrainingArguments(
    output_dir="./sentence_splitter_results",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=200,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_steps=480,
    seed=42,
)
```

## Project Structure

```text
├── Dataset.py
├── eval_results
    ├── confusion_matrix_sbd.png
├── utils.py
├── SenteceSplitter.ipynb
├── inference.py
├── baselineStandard.py
├── report.md
├── evaluation.md
```

## Setup

```bash
python -m venv SentenceSplitEnv
source SentenceSplitEnv/bin/activate
pip install torch transformers datasets nltk sentencepiece jupyter
```

## Run

### Build Unified Dataset

```bash
./SentenceSplitEnv/bin/python Dataset.py
```

### Train

Run the notebook:

```bash
jupyter notebook SenteceSplitter.ipynb
```

### Inference

```bash
./SentenceSplitEnv/bin/python inference.py
```

## Evaluation on Adversarial Test Set

Classification report for SentenceSplitterModel:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Word (0) | 0.9992 | 0.9759 | 0.9874 | 1244 |
| Sentence Boundary (1) | 0.8454 | 0.9939 | 0.9136 | 165 |
| Accuracy |  |  | 0.9780 | 1409 |
| Macro avg | 0.9223 | 0.9849 | 0.9505 | 1409 |
| Weighted avg | 0.9812 | 0.9780 | 0.9788 | 1409 |



## Hugging Face References 

- Model link: [SentenceSplitter-it-en](https://huggingface.co/LorenzoVentrone/SentenceSplitter-it-en)
- Dataset link: [LorenzoVentrone/SentenceSplitter-dataset](https://huggingface.co/datasets/LorenzoVentrone/SentenceSplitter-dataset)

## Base Model

- Backbone: [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
- Task: Token Classification (`num_labels = 2`)

## License

See `LICENSE` for project licensing details.
