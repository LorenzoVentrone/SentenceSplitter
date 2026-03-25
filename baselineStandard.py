import nltk
import spacy

nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

text = "Il Prof. Rossi è arrivato. Dopo la lezione ha incontrato l'A.D. della società S.p.a. a Roma."
sentence_nltk = sent_tokenize(text, language="italian")

print("--- NLTK ---")
for i, f in enumerate(sentence_nltk):
    print(f"{i+1}: {f}")

nlp = spacy.load("it_core_news_sm")

text = "Il Prof. Rossi è arrivato. Dopo la lezione ha incontrato l'A.D. della società S.p.a. a Roma."
doc = nlp(text)

print("\n--- spaCy ---")
sentence_spacy = [sent.text for sent in doc.sents]

for i, f in enumerate(sentence_spacy):
    print(f"{i+1}: {f}")