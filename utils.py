import tarfile
import json
from pathlib import Path
from datasets import load_dataset
import nltk

# Ensure NLTK punkt is downloaded for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# --- 2. TAR.GZ PROCESSING FUNCTION (Professor's Data) ---
def process_tar_dataset(file_path, window_size, stride, mode):
    """
    Reads a .tar.gz containing .sent_split files, extracts tokens based on <EOS>,
    and applies a sliding window per document.
    """
    chunks_tokens = []
    chunks_labels = []

    archive_path = Path(file_path)
    if not archive_path.exists():
        print(f"Warning: Archive {file_path} not found. Skipping.")
        return [], []

    print(f"Processing TAR archive: {archive_path.name}...")

    with tarfile.open(archive_path, "r:gz") as folder:
        for member in folder.getmembers():
            if member.isfile() and member.name.endswith(f"{mode}.sent_split"):
                f = folder.extractfile(member)
                if f is None:
                    continue

                text = f.read().decode("utf-8")
                doc_tokens = []
                doc_labels = []

                # Extract tokens and labels
                for word in text.split():
                    if "<EOS>" in word:
                        word_clean = word.replace("<EOS>", "")
                        if word_clean != "":
                            doc_tokens.append(word_clean)
                            doc_labels.append(1)
                    else:
                        doc_tokens.append(word)
                        doc_labels.append(0)

                # Apply sliding window for this document
                for i in range(0, len(doc_tokens), stride):
                    token_chunk = doc_tokens[i : i + window_size]
                    label_chunk = doc_labels[i : i + window_size]
                    chunks_tokens.append(token_chunk)
                    chunks_labels.append(label_chunk)

    return chunks_tokens, chunks_labels

# --- 3. JSONL PROCESSING FUNCTION (MultiLegalSBD Data) ---
def process_jsonl_dataset(file_path, window_size, stride):
    """
    Reads a JSONL dataset, extracts tokens and labels using 'token_end' spans,
    and applies a sliding window per document.
    """
    chunks_tokens = []
    chunks_labels = []
    
    path = Path(file_path)
    if not path.exists():
        print(f"Warning: File {file_path} not found. Skipping.")
        return [], []
        
    print(f"Processing JSONL dataset: {path.name}...")

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            json_line = json.loads(line)
            doc_tokens = []
            doc_labels = []
            tokens = json_line.get("tokens", [])

            # Keep only sentence spans and map token_end to the last non-whitespace token.
            eos_token_ids = set()
            for span in json_line.get("spans", []):
                if span.get("label") != "Sentence":
                    continue

                end_id = span.get("token_end")
                if end_id is None:
                    continue

                while end_id >= 0 and end_id < len(tokens):
                    t = tokens[end_id].get("text", "")
                    if t.strip() != "":
                        eos_token_ids.add(end_id)
                        break
                    end_id -= 1

            # Extract tokens and labels
            for token in tokens:
                token_text = token["text"]
                token_id = token["id"]
                
                # Skip whitespaces/newlines to match TAR processing logic
                if token_text.strip() == "":
                    continue
                    
                label = 1 if token_id in eos_token_ids else 0
                doc_tokens.append(token_text)
                doc_labels.append(label)

            # Apply Sliding Window for this document
            for i in range(0, len(doc_tokens), stride):
                token_chunk = doc_tokens[i : i + window_size]
                label_chunk = doc_labels[i : i + window_size]
                chunks_tokens.append(token_chunk)
                chunks_labels.append(label_chunk)

    return chunks_tokens, chunks_labels

# --- 4. WIKIPEDIA PROCESSING FUNCTION (Generalist Data) ---
def process_wikipedia_dataset(lang_config, num_articles, window_size, stride):
    """
    Downloads Wikipedia text, tokenizes it into sentences and words using NLTK,
    and applies the sliding window.
    """
    chunks_tokens = []
    chunks_labels = []
    
    print(f"Downloading Wikipedia ({lang_config}) - First {num_articles} articles...")
    dataset = load_dataset("wikimedia/wikipedia", lang_config, split=f"train[:{num_articles}]")
    
    # Optional: language mapping for NLTK
    nltk_lang = "italian" if "it" in lang_config else "english"

    for article in dataset:
        text = article["text"]
        if not text.strip():
            continue

        doc_tokens = []
        doc_labels = []

        # Split into paragraphs first (Wikipedia uses \n\n)
        paragraphs = text.split("\n\n")
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Use NLTK to find sentence boundaries
            sentences = nltk.sent_tokenize(para, language=nltk_lang)
            
            for sent in sentences:
                # Basic split to match our whitespace-based architecture
                words = sent.split()
                if not words:
                    continue
                
                for i, word in enumerate(words):
                    # If it's the last word of the NLTK sentence, label it 1
                    if i == len(words) - 1:
                        doc_tokens.append(word)
                        doc_labels.append(1)
                    else:
                        doc_tokens.append(word)
                        doc_labels.append(0)

        # Apply Sliding Window for this Wikipedia document
        for i in range(0, len(doc_tokens), stride):
            token_chunk = doc_tokens[i : i + window_size]
            label_chunk = doc_labels[i : i + window_size]
            chunks_tokens.append(token_chunk)
            chunks_labels.append(label_chunk)

    print(f"Extracted {len(chunks_tokens)} chunks from Wikipedia {lang_config}.")
    return chunks_tokens, chunks_labels