# NLP_demo

# Start Trial 
[Parts-of-speech.Info](http://parts-of-speech.info)
[Stanford parser](http://nlp.stanford.edu:8080/parser/index.jsp)

# Project 1: Spam detector

[spam data set](https://archive.ics.uci.edu/ml/datasets/Spambase)

# project 2: sentiment analysis
[sentiment data](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)

## Token 
Get the token from NLTK
```python
from nltk.stem import WordNetLemmatizer
tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
#Change into basic format: jumping->jump dogs->dog
wordnet_lemmatizer = WordNetLemmatizer()
#remove stop words
tokens = [t for t in tokens if t not in stopwords]
```

## nltk download
need to download nltk package besides just install it, based on the run time error, install the missing 
```python
# First start ipython or python terminal
import nltk
nltk.download()
```
this command will start the GUI for NLTK package download, select the missing ones


# Python 3 change

1. use dict.items() instead of dict.iteritems()


