import tarfile
from pathlib import Path
from datasets import Dataset


dataset_chunks_tokens = []
dataset_chunks_labels = []

window_size = 128
stride = 100

percorso_archivio = Path("sent_split_data.tar.gz")

with tarfile.open(percorso_archivio, "r:gz") as folder:
    for file in folder.getmembers():
        
        if file.isfile() and file.name.endswith(".sent_split"):
            print(f"Elaborazione di: {file.name}...")

            f = folder.extractfile(file)
            if f is None:
                continue

            text = f.read().decode("utf-8")

            file_tokens = []
            file_labels = []

            for word in text.split():
                if "<EOS>" in word:
                    word_clean = word.replace("<EOS>", "")
                    if word_clean != "":
                        file_tokens.append(word_clean)
                        file_labels.append(1)
                else:
                    file_tokens.append(word)
                    file_labels.append(0)

            for i in range(0, len(file_tokens), stride):
                token_chunk = file_tokens[i : i + window_size]
                label_chunk = file_labels[i : i + window_size]
                
                dataset_chunks_tokens.append(token_chunk)
                dataset_chunks_labels.append(label_chunk)

print("\n--- Estrazione Completata ---")
print(f"Totale chunk creati: {len(dataset_chunks_tokens)}")

# 1. Creiamo l'oggetto Dataset nativo di Hugging Face
hf_dataset = Dataset.from_dict({
    "tokens": dataset_chunks_tokens,
    "ner_tags": dataset_chunks_labels
})

# 2. Salviamo il dataset in una cartella locale
hf_dataset.save_to_disk("ReadyDataset")

print("Dataset salvato con successo!")
