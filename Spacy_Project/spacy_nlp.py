import spacy                         # See "Installing spaCy"
import numpy
nlp = spacy.load('en')               # You are here.
doc = nlp(u'Last week, Mobileye (NYSE:MBLY) announced that it was going to be acquired by Intel (NASDAQ:INTC) in a mammoth $15.3 billion deal.')          # See "Using the pipeline"


#print((w.text, w.pos_) for w in doc)
for proc in nlp.pipeline:
    proc(doc)
    print(doc.vector.shape)

for word in doc:
    print(word.text, word.lemma, word.lemma_, word.tag, word.tag_, word.pos, word.pos_)



doc.ents = []