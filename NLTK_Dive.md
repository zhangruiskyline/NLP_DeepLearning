#Part 1: Getting Started with NLTK

Referring to [dive to NLTK](http://textminingonline.com/dive-into-nltk-part-i-getting-started-with-nltk)

##ABout NLTK

Here is a description from the NLTK official site:

 > NLTK is a leading platform for building Python programs to work with human language data. 
 It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, 
 along with a suite of text processing libraries for classification, tokenization, 
 stemming, tagging, parsing, and semantic reasoning.
 
##Installtion

```shell
#install package
sudo pip install -U pyyaml nltk
```
###Download NLTK data
Installing NLTK Data
After installing NLTK, you need install NLTK Data which include a lot of corpora, grammars, models and etc. Without NLTK Data, NLTK is nothing. You can find the complete nltk data list here: 
[NLTK data](http://nltk.org/nltk_data/)

```python
#GUI downloader
import nltk
nltk.download()
#command download could be
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
```

##Test 
* 1) Test Brown Corpus
```python
from nltk.corpus import brown
brown.words()[0:10]
len(brown.words())
dir(brown)

```

* 2) Test NLTK Book Resources:
```python
from nltk.book import *
dir(text1)
```

* 3) Sent Tokenize(sentence boundary detection, sentence segmentation), Word Tokenize and Pos Tagging:
```python
from nltk import sent_tokenize, word_tokenize, pos_tag
text = "Machine learning is the science of getting computers to act without being explicitly programmed. In the past decade, machine learning has given us self-driving cars, practical speech recognition, effective web search, and a vastly improved understanding of the human genome. Machine learning is so pervasive today that you probably use it dozens of times a day without knowing it. Many researchers also think it is the best way to make progress towards human-level AI. In this class, you will learn about the most effective machine learning techniques, and gain practice implementing them and getting them to work for yourself. More importantly, you'll learn about not only the theoretical underpinnings of learning, but also gain the practical know-how needed to quickly and powerfully apply these techniques to new problems. Finally, you'll learn about some of Silicon Valley's best practices in innovation as it pertains to machine learning and AI."
sents = sent_tokenize(text)
tokens = word_tokenize(text)
tagged_tokens = pos_tag(tokens)
```

# Part II: Sentence Tokenize and Word Tokenize
##sentence token
sent_tokenize uses an instance of PunktSentenceTokenizer from the nltk. 
tokenize.punkt module. This instance has already been trained on and works well for many European languages. 
So it knows what punctuation and characters mark the end of a sentence and the beginning of a new sentence.

sent_tokenize is one of instances of PunktSentenceTokenizer from the nltk.tokenize.punkt module. 
Tokenize Punkt module has many pre-trained tokenize model for many european languages, here is the list from the
nltk_data/tokenizers/punkt/README file:

```python
text = "this is a sent tokenize test. this is sent two. is this sent three? sent 4 is cool! Now it’s your turn."
from nltk.tokenize import sent_tokenize
sent_tokenize_list = sent_tokenize(text)
len(sent_tokenize_list)
#out:5
sent_tokenize_list
['this is a sent tokenize test.',
 'this is sent two.',
 'is this sent three?',
 'sent 4 is cool!',
 'Now it’s your turn.']
 
import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer.tokenize(text)
['this is a sent tokenize test.',
 'this is sent two.',
 'is this sent three?',
 'sent 4 is cool!',
 'Now it’s your turn.']
```
##Tokenizing text into words
Tokenizing text into words in NLTK is very simple, just called word_tokenize from nltk.tokenize module:
```python
from nltk.tokenize import word_tokenize
word_tokenize("this’s a test")
#Out: 
['this’s', 'a', 'test']
##Actually, word_tokenize is a wrapper function that calls tokenize by the TreebankWordTokenizer
# This below is same
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
tokenizer.tokenize("this’s a test")
#Out: 
['this’s', 'a', 'test']
```
Except the TreebankWordTokenizer, 
there are other alternative word tokenizers, such as **PunktWordTokenizer** and **WordPunktTokenizer**.
>PunktTokenizer splits on punctuation, but keeps it with the word:
```python
## Newest nltk versions > 3.0.0 do not have PunktWordTokenizer class any more.
from nltk.tokenize import PunktWordTokenizer
punkt_word_tokenizer.tokenize("this’s a test”)
['this’, “‘s”, ‘a’, ‘test']
```
>WordPunctTokenizer splits all punctuations into separate tokens:
```python
from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_punct_tokenizer.tokenize("This’s a test")
#Out
['This', '’', 's', 'a', 'test']

```
