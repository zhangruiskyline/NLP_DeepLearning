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

# Part 2: Sentence Tokenize and Word Tokenize

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

# Part 3: Part-Of-Speech Tagging and POS Tagger

Part-of-speech tagging is one of the most important text analysis tasks used to classify words 
into their part-of-speech and label them according the tagset which is a collection of tags used for the pos tagging. 
Part-of-speech tagging also known as word classes or lexical categories.

```python
import nltk
text = nltk.word_tokenize("Dive into NLTK: Part-of-speech tagging and POS Tagger")
nltk.pos_tag(text)
[('Dive', 'NNP'),
 ('into', 'IN'),
 ('NLTK', 'NNP'),
 (':', ':'),
 ('Part-of-speech', 'JJ'),
 ('tagging', 'NN'),
 ('and', 'CC'),
 ('POS', 'NNP'),
 ('Tagger', 'NNP')]
```
NLTK provides documentation for each tag, which can be queried using the tag, e.g., 
nltk.help.upenn_tagset(‘RB’), or a regular expression, e.g., nltk.help.upenn_brown_tagset(‘NN.*’):
[pos tag link](http://textanalysisonline.com/nltk-pos-tagging)

The default pos tagger model using in NLTK is maxent_treebanck_pos_tagger model, 
you can find the code in *nltk-master/nltk/tag/__init__.py*

You can find the pre-trained POS Tagging Model in __nltk_data/taggers__

How to train a POS Tagging Model or POS Tagger in NLTK
You have used the maxent treebank pos tagging model in NLTK by default, 
and NLTK provides not only the maxent pos tagger, but other pos taggers 
like crf, hmm, brill, tnt and interfaces with stanford pos tagger, hunpos pos tagger and senna postaggers

```
-rwxr-xr-x@ 4.4K 7 22 2013 __init__.py
-rwxr-xr-x@ 2.9K 7 22 2013 api.py
-rwxr-xr-x@ 56K 7 22 2013 brill.py
-rwxr-xr-x@ 31K 7 22 2013 crf.py
-rwxr-xr-x@ 48K 7 22 2013 hmm.py
-rwxr-xr-x@ 5.1K 7 22 2013 hunpos.py
-rwxr-xr-x@ 11K 7 22 2013 senna.py
-rwxr-xr-x@ 26K 7 22 2013 sequential.py
-rwxr-xr-x@ 3.3K 7 22 2013 simplify.py
-rwxr-xr-x@ 6.4K 7 22 2013 stanford.py
-rwxr-xr-x@ 18K 7 22 2013 tnt.py
-rwxr-xr-x@ 2.3K 7 22 2013 util.py
```

## Example to train TnT POS Tagger Model
```python
from nltk.corpus import treebank
len(treebank.tagged_sents())
train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:]

'''
We use the first 3000 treebank tagged sentences as the train_data, and last 914 tagged sentences as the test_data, 
now we train TnT POS Tagger by the train_data and evaluate it by the test_data:
'''

from nltk.tag import tnt
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(train_data)
tnt_pos_tagger.evaluate(test_data)

## we can save
import pickle
f = open('tnt_treebank_pos_tagger.pickle', 'w')
pickle.dump(tnt_pos_tagger, f)
f.close()

#can use it any time you want:
tnt_tagger.tag(nltk.word_tokenize("this is a tnt treebank tnt tagger"))
```

# Part 4: Stemming and Lemmatization

Stemming and Lemmatization are the basic text processing methods for English text. 
The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form. 
Here is the definition from wikipedia for stemming and lemmatization:

## Stemmer in NLTK
NLTK provides several famous stemmers interfaces, such as 
Porter stemmer, Lancaster Stemmer, Snowball Stemmer and etc. In NLTK, using those stemmers is very simple.

For Porter Stemmer, which is based on The Porter Stemming Algorithm, can be used like this:

## How to use Lemmatizer in NLTK
The NLTK Lemmatization method is based on WordNet’s built-in morphy function. 
You would note that the “are” and “is” lemmatize results are not “be”, 
that’s because the lemmatize method default pos argument is “n”: we can change to

```
lemmatize(word, pos=’n’)
```

Examples can be found in [stemming-and-lemmatization](http://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization)

# Part 5 Using Stanford Text Analysis Tools in Python
NLTK now provides interfaces for 
  * [Stanford Part-Of-Speech Tagger (POS)](http://nlp.stanford.edu/software/tagger.shtml)
  * [Stanford Named Entity Recognizer (NER)](http://nlp.stanford.edu/software/CRF-NER.shtml)  
  * [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml)
  * [Stanford Segmenter](http://nlp.stanford.edu/software/segmenter.shtml)

## POS Tagger
Stanford POS Tagger official site provides two versions of POS Tagger:
We suggest you download the full version which contains a lot of models.
After downloading the full version, unzip it and copy the related data in our test directory:
```python
import os
##set OS path to Stanford postagger
os.environ['STANFORD_POSTAGGER'] = "/Users/ruizhang/Documents/stanford-postagger-full-2016-10-31"
from nltk.tag.stanford import StanfordPOSTagger
english_postagger = StanfordPOSTagger(os.environ['STANFORD_POSTAGGER'] +'/models/english-bidirectional-distsim.tagger',os.environ['STANFORD_POSTAGGER']+'/stanford-postagger.jar')
english_postagger.tag('this is stanford postagger in nltk for python users'.split())
#Out: 
[('this', 'DT'),
 ('is', 'VBZ'),
 ('stanford', 'JJ'),
 ('postagger', 'NN'),
 ('in', 'IN'),
 ('nltk', 'NN'),
 ('for', 'IN'),
 ('python', 'NN'),
 ('users', 'NNS')]
 
## And it supports multiple languages
chinese_postagger = StanfordPOSTagger(os.environ['STANFORD_POSTAGGER'] +'/models/chinese-distsim.tagger',os.environ['STANFORD_POSTAGGER']+'/stanford-postagger.jar',encoding='utf-8')
chinese_postagger.tag('这 是 在 Python 环境 中 使用 斯坦福 词性 标 器'.split())
#Out:
[('', '这#PN'),
 ('', '是#VC'),
 ('', '在#P'),
 ('', 'Python#NN'),
 ('', '环境#NN'),
 ('', '中#LC'),
 ('', '使用#VV'),
 ('', '斯坦福#NR'),
 ('', '词性#JJ'),
 ('', '标#NN'),
 ('', '器#NN')]
```
The models contains a lot of pos tagger models, you can find the details info from the README-Models.txt

## NER
Named Entity Recognition (NER) labels sequences of words in a text which are the names of things, 
such as person and company names, or gene and protein names. It comes with well-engineered feature extractors for Named Entity Recognition, 
and many options for defining feature extractors. Included with the download are good named entity recognizers for English,
particularly for the 3 classes (PERSON, ORGANIZATION, LOCATION), and we also make available on this page various other models for different languages and circumstances,
including models trained on just the CoNLL 2003 English training data. The distributional similarity features in some models improve performance 
but the models require considerably more memory.

  * First step is to download NER from official website [Stanford NER](http://nlp.stanford.edu/software/CRF-NER.shtml#Download)
  * Use it similar in python
```python
import os
from nltk.tag.stanford import StanfordNERTagger
os.environ['STANFORD_NER'] = "/Users/ruizhang/Documents/stanford-ner-2016-10-31"
english_nertagger = StanfordNERTagger(os.environ['STANFORD_NER']+'/classifiers/english.all.3class.distsim.crf.ser.gz', os.environ['STANFORD_NER']+'/stanford-ner.jar')
english_nertagger.tag('Rui Zhang is working in San Francisco Bay Area in Broadcom as a software engineer'.split())
#Out[61]
[('Rui', 'PERSON'),
 ('Zhang', 'PERSON'),
 ('is', 'O'),
 ('working', 'O'),
 ('in', 'O'),
 ('San', 'LOCATION'),
 ('Francisco', 'LOCATION'),
 ('Bay', 'LOCATION'),
 ('Area', 'LOCATION'),
 ('in', 'O'),
 ('Broadcom', 'ORGANIZATION'),
 ('as', 'O'),
 ('a', 'O'),
 ('software', 'O'),
 ('engineer', 'O')]
```
The Models Included with Stanford NER are a 4 class model trained for CoNLL, a 7 class model trained for MUC, and a 3 class model trained on both data sets for the intersection of those class sets.

 >3 class:	Location, Person, Organization
 >4 class:	Location, Person, Organization, Misc
 >7 class:	Time, Location, Organization, Person, Money, Percent, Date

You can test the 7 class Stanford NER on  [Text Analysis Online Demo](http://textanalysisonline.com)

##Parser
A natural language parser is a program that works out the grammatical structure of sentences, for instance,
which groups of words go together (as “phrases”) and which words are the subject or object of a verb. 
Probabilistic parsers use knowledge of language gained from hand-parsed sentences to try to produce the most likely analysis of new sentences. 
These statistical parsers still make some mistakes, 
but commonly work rather well. Their development was one of the biggest breakthroughs in natural language processing in the 1990s.

  * First step is to download parser from official website [Stanford parser](http://nlp.stanford.edu/software/lex-parser.shtml#Download)
  * Use it similar in python
```python
import os
from nltk.parse.stanford import StanfordParser
os.environ['STANFORD_PARSER'] = "/Users/ruizhang/Documents/stanford-parser-full-2016-10-31"
english_parser = StanfordParser(os.environ['STANFORD_PARSER'] + 'stanford-parser.jar', os.environ['STANFORD_PARSER'] + 'stanford-parser-3.7.0-models.jar')
english_parser.raw_parse_sents(("this is the english parser test”, “the parser is from stanford parser"))
#Out:
```

## Stanford Word Segmenter
* download [Stanford Word Segmenter](http://nlp.stanford.edu/software/segmenter.shtml#Download)
* Python interface:
```python
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
segmenter = StanfordSegmenter(path_to_jar=sg+"/stanford-segmenter-3.7.0.jar", path_to_slf4j='/Users/ruizhang/Documents/slf4j-1.7.23/slf4j-log4j12-1.7.23.jar',path_to_sihan_corpora_dict=sg+"/data", path_to_model=sg+"/data/pku.gz", path_to_dict=sg+"/data/dict-chris6.ser.gz")
sentence = u"这是中文分词测试"
segmenter.segment(sentence)
#Out: 
'这 是 中文 分词 测试\n'
```

# Part 6:  Text Classification
