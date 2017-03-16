import spacy                         # See "Installing spaCy"
nlp = spacy.load('en')               # You are here.
doc = nlp(u'Apple needs to make an acquisition to revive its story')          # See "Using the pipeline"
print((w.text, w.pos_) for w in doc)
for proc in nlp.pipeline:
    proc(doc)

for word in doc:
    print(word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_)

doc.ents = []