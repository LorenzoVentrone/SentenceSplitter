# NLP Sentence Splitter Project Report

## 1. Introduction

This project builds a multilingual sentence boundary detection system as a token classification task. Instead of relying on regex-based splitting, the model predicts sentence boundaries directly on tokens:

- 0: token is not the end of a sentence
- 1: token is the end of a sentence

The current version uses a unified training pipeline combining academic, legal, and general-domain data, while enforcing train-only filtering for sources that provide split files.

## 2. Dataset Construction (Current Version)

Dataset creation is orchestrated by `Dataset.py` and helper functions in `utils.py`.

### 2.1 Data Sources

The final unified dataset is built from three sources:

1. Professor dataset from `sent_split_data.tar.gz` (only `*-train.sent_split` files)
2. MultiLegalSBD datasets from `MultiLegalSBD/*train.jsonl`
3. Wikipedia general-domain data:
   - Italian: `20231101.it`
   - English: `20231101.en`

### 2.2 TAR Processing

For professor `.sent_split` files:

- Tokens are extracted with whitespace split
- Tokens containing `<EOS>` are cleaned and labeled `1`
- All other tokens are labeled `0`
- Only files ending with `-train.sent_split` are included
- Sliding-window chunking is applied per document

### 2.3 MultiLegalSBD JSONL Processing

For legal JSONL files:

- Input fields: `tokens` and `spans`
- Only spans with `label == "Sentence"` are used
- `token_end` is mapped to the last non-whitespace token
- Whitespace-only tokens are skipped in output
- Sliding-window chunking is applied

### 2.4 Wikipedia Integration

Wikipedia articles are downloaded with Hugging Face datasets and processed as follows:

- Paragraph split by blank lines
- Sentence detection with NLTK `sent_tokenize`
- Tokenization by whitespace split
- Last token of each sentence gets label `1`
- Sliding-window chunking is applied

Current setting:

- `WIKI_ARTICLES_PER_LANG = 125`

### 2.5 Unified Dataset Output

After concatenation, data is exported as a Hugging Face dataset with:

- `tokens`
- `ner_tags`

Saved folder:

- `unified_training_dataset`

## 3. Training Setup (Current Version)

Training is run in `SenteceSplitter.ipynb` with `xlm-roberta-base` as backbone (`num_labels = 2`).

### 3.1 Split Used in Latest Run

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

Note: this `test` split is used as validation during training.

### 3.2 Training Hyperparameters

```python
TrainingArguments(
    output_dir="./risultati_sentence_splitter",
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

## 4. Inference Pipeline

Inference is implemented in `inference.py`:

1. Load tokenizer and fine-tuned model
2. Split long input into manageable blocks
3. Predict token-level EOS labels
4. Reconstruct final sentences from predicted boundaries

## 5. Version Summary

Current version updates:

1. Train-only professor files (`-train.sent_split`)
2. Train-only legal files (`*train.jsonl`)
3. Unified output in `unified_training_dataset`
4. Current split: 19,229 train / 2,137 validation
5. Updated training configuration with `warmup_steps=480`

## 6. Hugging Face Release

The model can be published to Hugging Face Hub.

Model URL: `<ADD_HUGGING_FACE_MODEL_LINK_HERE>`

## 7. Conclusion

The current pipeline standardizes data from multiple sources while reducing split leakage risk through train-only source filtering for professor and legal corpora. This setup is intended to improve robustness while preserving a clean evaluation protocol on held-out test sets.
