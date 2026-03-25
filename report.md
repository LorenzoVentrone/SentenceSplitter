# NLP Sentence Splitter Project Report

## 1. Introduction

This project builds a multilingual sentence boundary detection system as a token classification task. Instead of relying on regular-expression sentence splitting, the model predicts sentence boundaries directly on tokens, with labels:

- 0: token is not the end of a sentence
- 1: token is the end of a sentence

The latest version extends the original pipeline by integrating multiple heterogeneous sources (academic annotation, legal JSONL corpora, and Wikipedia), unifying them into one training dataset, and training a fine-tuned XLM-RoBERTa model for robust behavior on domain-specific and general text.

## 2. Dataset Construction (Latest Version)

Dataset creation is orchestrated by `Dataset.py` and helper functions in `utils.py`.

### 2.1 Data Sources

The final unified dataset is built from three sources:

1. Professor dataset from `sent_split_data.tar.gz`
2. MultiLegalSBD datasets from all `*.jsonl` files in the `MultiLegalSBD` folder
3. Wikipedia general-domain data:
   - Italian: `20231101.it`
   - English: `20231101.en`

### 2.2 TAR Processing

For `.sent_split` files:

- Tokens are extracted by whitespace split
- Tokens containing `<EOS>` are cleaned and labeled as `1`
- All other tokens are labeled as `0`
- Sliding-window chunking is applied per document

### 2.3 MultiLegalSBD JSONL Processing

The JSONL pipeline was updated for stronger compatibility and cleaner supervision:

- Reads records containing `tokens` and `spans`
- Uses only spans with `label == "Sentence"`
- Uses `token_end` as sentence boundary target
- If `token_end` points to whitespace/newline, it backtracks to the last non-whitespace token
- Skips whitespace-only tokens in the final token stream
- Applies the same sliding-window chunking strategy used by TAR data

This ensures consistent token/label alignment across legal corpora with tokenized annotations.

### 2.4 Wikipedia Integration

Wikipedia articles are downloaded with the Hugging Face datasets library and processed as follows:

- Text is split into paragraphs
- Sentence boundaries are identified with NLTK sentence tokenizer (`sent_tokenize`)
- Tokens are produced via whitespace split to keep tokenization style consistent across sources
- Last token of each detected sentence receives label `1`
- Sliding-window chunking is applied

In this version, `WIKI_ARTICLES_PER_LANG` is set to 250.

### 2.5 Unified Dataset Output

After concatenating all sources, the pipeline creates a Hugging Face dataset with fields:

- `tokens`
- `ner_tags`

and saves it to:

- `unified_training_dataset`

## 3. Training Setup

Training is performed in `SenteceSplitter.ipynb` using `xlm-roberta-base` as backbone for token classification (`num_labels = 2`).

Main steps:

1. Load unified dataset
2. Build train/test split
3. Tokenize and align word-level labels with subword tokens
4. Ignore non-first subword labels via `-100`
5. Train with Hugging Face Trainer API

### 3.1 Final Split (Latest Run)

```python
DatasetDict({
    train: Dataset({
        features: ['tokens', 'ner_tags'],
        num_rows: 27720
    })
    test: Dataset({
        features: ['tokens', 'ner_tags'],
        num_rows: 3081
    })
})
```

## 4. Inference Pipeline

Inference is executed through `inference.py`.

High-level flow:

1. Load tokenizer and fine-tuned model
2. Split long input text into manageable blocks
3. Predict token-level EOS labels
4. Reconstruct final sentences from predicted boundaries

The script is designed to handle long and noisy real-world text while preserving sentence segmentation quality.

## 5. Version Updates Summary

Compared to the previous version, the following major updates were introduced:

1. Multi-source dataset merging (Professor + MultiLegalSBD + Wikipedia IT/EN)
2. Automatic discovery of legal JSONL files from `MultiLegalSBD/*.jsonl`
3. Improved JSONL boundary extraction logic for whitespace-safe `token_end` handling
4. Unified output dataset folder (`unified_training_dataset`)
5. Final expanded training split with 27,720 training rows and 3,081 test rows
6. Model publication to Hugging Face Hub

## 6. Hugging Face Release

The trained model is available on Hugging Face Hub.

Model URL: `<ADD_HUGGING_FACE_MODEL_LINK_HERE>`

## 7. Conclusion

The latest pipeline version improves both coverage and robustness by combining legal-domain and general-domain data under a single consistent labeling strategy. This update significantly strengthens training diversity and supports better sentence boundary detection across different text genres.
