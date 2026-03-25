import nltk
# Scarica i dati pre-addestrati (da eseguire solo una volta)
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

testo = "Il Prof. Rossi è arrivato. Dopo la lezione ha incontrato l'A.D. della società S.p.a. a Roma."
# È fondamentale specificare la lingua per caricare il modello giusto
frasi_nltk = sent_tokenize(testo, language="italian")

print("--- NLTK ---")
for i, f in enumerate(frasi_nltk):
    print(f"{i+1}: {f}")


import spacy

# Devi scaricare il modello da riga di comando prima di eseguire lo script:
# python -m spacy download it_core_news_sm

# Carica il modello italiano
nlp = spacy.load("it_core_news_sm")

testo = "Il Prof. Rossi è arrivato. Dopo la lezione ha incontrato l'A.D. della società S.p.a. a Roma."
doc = nlp(testo)

print("\n--- spaCy ---")
# Estrae le frasi tramite l'attributo .sents
frasi_spacy = [sent.text for sent in doc.sents]

for i, f in enumerate(frasi_spacy):
    print(f"{i+1}: {f}")