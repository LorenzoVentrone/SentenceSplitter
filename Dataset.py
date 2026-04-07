from datasets import Dataset
from pathlib import Path
import utils

# --- 1. CONFIGURATION ---
WINDOW_SIZE = 128
STRIDE = 100
WIKI_ARTICLES_PER_LANG = 125 # Number of Wikipedia articles to download per language

# --- MAIN EXECUTION PIPELINE ---
if __name__ == "__main__":
    print("=== STARTING DATA EXTRACTION PIPELINE ===\n")

    # Process Professor's Data
    prof_tokens, prof_labels = utils.process_tar_dataset(
        "sent_split_data.tar.gz", WINDOW_SIZE, STRIDE, "train"
    )

    # Process all MultiLegalSBD JSONL files
    legal_tokens = []
    legal_labels = []
    legal_files = sorted(Path("MultiLegalSBD").rglob("*train.jsonl"))

    if not legal_files:
        print("Warning: No JSONL files found in ./MultiLegalSBD")
    else:
        print(f"Found {len(legal_files)} legal JSONL files.")

    for jsonl_file in legal_files:
        file_tokens, file_labels = utils.process_jsonl_dataset(
            str(jsonl_file), WINDOW_SIZE, STRIDE
        )
        legal_tokens.extend(file_tokens)
        legal_labels.extend(file_labels)

    # Process Generalist Data (Wikipedia IT & EN)
    print("\n--- Processing Generalist Datasets ---")
    wiki_it_tokens, wiki_it_labels = utils.process_wikipedia_dataset(
        "20231101.it", WIKI_ARTICLES_PER_LANG, WINDOW_SIZE, STRIDE
    )
    
    wiki_en_tokens, wiki_en_labels = utils.process_wikipedia_dataset(
        "20231101.en", WIKI_ARTICLES_PER_LANG, WINDOW_SIZE, STRIDE
    )

    # Merge all datasets
    all_tokens = prof_tokens + legal_tokens + wiki_it_tokens + wiki_en_tokens
    all_labels = prof_labels + legal_labels + wiki_it_labels + wiki_en_labels

    print(f"\n=== MERGE COMPLETE ===")
    print(f"Professor chunks: {len(prof_tokens)}")
    print(f"Legal chunks: {len(legal_tokens)}")
    print(f"Wikipedia chunks: {len(wiki_it_tokens) + len(wiki_en_tokens)}")
    print(f"Total chunks ready: {len(all_tokens)}")
    
    # Sanity check
    assert len(all_tokens) == len(all_labels), "Mismatch between tokens and labels lists!"

    # --- HUGGING FACE DATASET CREATION ---
    print("\nConverting to Hugging Face Dataset format...")
    hf_dataset = Dataset.from_dict({
        "tokens": all_tokens,
        "ner_tags": all_labels
    })

    # Save to disk in Arrow/Parquet format
    output_folder = "unified_training_dataset"
    hf_dataset.save_to_disk(output_folder)

    print(f"Success! Dataset saved to './{output_folder}'.")
    