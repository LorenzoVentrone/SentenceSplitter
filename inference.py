import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load model and tokenizer
model_path = "/Users/lorenzoventrone/Desktop/NLP/sentenceSplitter/SentenceSplitterModel"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
text_path = "tulu3.txt"

# with open(text_path, "r", encoding="utf-8") as f:
#     secret_text = f.read() 

secret_text = """
Verbale di Assemblea Straordinaria - Q4

Il giorno 12.10.2023, alle ore 14.30, presso la sede legale della Future Tech S.p.A., si è riunito il consiglio di amministrazione. L'A.D. della società, Ing. A. B. Rossi, ha preso la parola per primo. "I ricavi netti sono aumentati del 4.25% (pari a circa 1.8 mln di euro) rispetto al trimestre precedente." ha affermato con soddisfazione il Dott. Verdi. Tuttavia... restano alcune criticità da affrontare sul mercato estero. 

Secondo l'art. 4, comma 2.1 dello statuto aziendale, le decisioni straordinarie richiedono la maggioranza assoluta. La Dott.ssa Neri, V.P. delle vendite, concorda pienamente (seppur con alcune riserve tecniche, n.d.r.). Si vedano a tal proposito i documenti allegati al fascicolo: all. A, all. B, all. C, ecc. Il budget previsto per l'espansione negli U.S.A. e nell'U.E. deve essere categoricamente rivisto prima di fine anno.

Analisi Scientifica e Prospettive

Come citato dal Prof. J. K. R. Tolkien nel suo recente paper "AI, NLP e modelli generativi: Miti e Realtà" (pubblicato integralmente sul sito www.ricerca-tech.it/nlp-news), il tasso di errore dei Transformer è sceso allo 0.05%. L'autore scrive testualmente: "L'implementazione di reti neurali profonde (e.g., XLM-RoBERTa) cambierà drasticamente il paradigma dell'estrazione dati.". Questa affermazione ha scosso l'intera comunità scientifica. Per ulteriori informazioni tecniche è possibile contattare l'indirizzo mario.rossi@universita.edu.it oppure chiamare il numero verde 800.123.456 entro venerdì p.v.

Ordine del Giorno

1. Approvazione del bilancio consuntivo.
2. Nomina del nuovo C.T.O. aziendale.
3. Varie ed eventuali.

N.B. Il punto 3 include una discussione riservata sul progetto "Leonardo Cineca". P.S. Si ricorda ai signori soci (cfr. la mail inviata il 10 u.s.) di portare i badge magnetici. La seduta è tolta.
"""

print("---INPUT---")
print(secret_text)

# Split the text into paragraphs to avoid Transformer's max token limit (512)
paragraphs = secret_text.split("\n\n")

# Logic override: acronyms that should never trigger an EOS
protected_acronyms = ["P.S.", "N.B."]
final_sentences = []

# Perform inference paragraph by paragraph
for paragraph in paragraphs:
    words = paragraph.split()
    
    if len(words) == 0:
        continue # Skip empty paragraphs
        
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Safely handle dimensionality for logits and predictions
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0].tolist()
        
    word_ids = inputs.word_ids() 

    current_sentence = []
    prev_word_idx = None

    # Reconstruct sentences from sub-words
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue 

        if word_idx != prev_word_idx:
            original_word = words[word_idx]
            current_sentence.append(original_word)

            # Cut the sentence if model predicts 1 AND word is not protected
            if predictions[idx] == 1 and original_word.upper() not in protected_acronyms:
                final_sentences.append(" ".join(current_sentence))
                current_sentence = [] # Empty the buffer for the next sentence

            prev_word_idx = word_idx

    # Catch any remaining words at the end of the paragraph
    if current_sentence:
        final_sentences.append(" ".join(current_sentence))

# --- PRINT RESULTS ---
print("\n--- MODEL OUTPUT ---")
for i, sentence in enumerate(final_sentences):
    print(f"[{i+1}] {sentence}")