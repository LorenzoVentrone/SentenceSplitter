# NLP Sentence Splitter Project Report

## 1. Introduction
This project focuses on building a robust sentence boundary detection (sentence splitting) system using a token classification approach. Utilizing data provided by the NLP instructor, the project involves creating a custom dataset, training an XLM-RoBERTa model, and deploying the model to perform inference on unseen text. The pipeline consists of three main phases: Dataset Creation, Model Training, and Inference.

## 2. Dataset Creation
The dataset preparation is handled by the `Dataset.py` script. The raw data is provided in a compressed archive (`sent_split_data.tar.gz`). 

**Preprocessing Steps:**
- **Parsing:** The script iterates through `.sent_split` files within the archive.
- **Label Extraction:** The text is split into words. The script searches for the `<EOS>` (End Of Sentence) tag attached to words. 
  - Words containing `<EOS>` have the tag stripped and are assigned a label of `1` (indicating a sentence boundary).
  - All other words are assigned a label of `0`.
- **Chunking:** To accommodate the maximum sequence length constraints of Transformer models, the parsed tokens and labels are divided into overlapping chunks. A `window_size` of 128 tokens and a `stride` of 100 tokens are used to maintain contextual continuity across chunks.
- **Dataset Generation:** A native Hugging Face `Dataset` object is created containing the `tokens` and their corresponding `ner_tags`. The dataset is finally saved locally as `ReadyDataset`.

## 3. Model Training
The training phase is documented in the Jupyter Notebook `SenteceSplitter.ipynb`. The goal is to fine-tune a pre-trained language model on the custom sentence splitting dataset.

**Training Workflow:**
- **Data Loading & Splitting:** The previously saved `ReadyDataset` is loaded and split into a training set (90%) and a test/validation set (10%).
- **Model Selection:** The project utilizes the `xlm-roberta-base` model, which provides strong multilingual representations.
- **Tokenization and Alignment:** Because RoBERTa uses sub-word tokenization (SentencePiece/BPE), a crucial step is aligning the original word labels with the newly generated sub-words. The first sub-word of an original word receives the true label (`0` or `1`), while subsequent sub-words are assigned a label of `-100` so they are ignored during the loss calculation.
- **Hyperparameters:** The model is trained using the `Trainer` API with a sequence classification head (`num_labels=2`). The configurations used are:
  - **Epochs:** 3
  - **Learning Rate:** 2e-5
  - **Batch Size:** 16 (for both training and evaluation)
  - **Weight Decay:** 0.01
- After 3 epochs of training (yielding a steady decrease in both training and validation loss), the trained model and tokenizer are saved to the `SentenceSplitterModel` directory.

## 4. Inference and Results
To evaluate the model's real-world performance, inference is performed on a "secret text" (e.g., `tulu3.txt` or a complex Italian corporate report) using the `inference.py` script.

**Inference Mechanism:**
- **Paragraph Splitting:** To avoid exceeding the Transformer's token limit (512 tokens), the input text is first divided by paragraphs.
- **Prediction:** Each paragraph is tokenized and fed to the fine-tuned XLM-RoBERTa model to retrieve logits. `torch.argmax` is used to predict the class (`0` or `1`) of each sub-word.
- **Reconstruction:** The script maps the sub-word predictions back to the original words. 
- **Rule-based Override:** A manual override is included for specific protected acronyms (like `"P.S."` and `"N.B."`) which historically confuse sentence splitters. Even if the model predicts an EOS for these tokens, the script forces the sentence to continue.
- **Output:** The final sentences are aggregated and printed clearly. 

The inference script validates that the customized model alongside heuristic overrides successfully segments complex, real-world text into coherent single sentences.
