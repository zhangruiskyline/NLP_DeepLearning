# NLP_demo

# Start Trial 
  1. [Parts-of-speech.Info](http://parts-of-speech.info)

  2. [Stanford parser](http://nlp.stanford.edu:8080/parser/index.jsp)

# Project 1: Spam detector

[spam data set](https://archive.ics.uci.edu/ml/datasets/Spambase)

# project 2: sentiment analysis
[sentiment data](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)

# NLTK

A example for all usage:
[textanalysis API](https://market.mashape.com/textanalysis/textanalysis#nltk-wordnet-word-lemmatizer)

## nltk download corresponding packages
need to download nltk package besides just install it, based on the run time error, install the missing 
```python
# First start ipython or python terminal
import nltk
nltk.download()
```
this command will start the GUI for NLTK package download, select the missing ones

## Pos Tag
```python
nltk.pos_tag("This is a test string".split())
#output will be
[('this', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('test', 'NN'), ('string', 'NN')]
```
for all the pos tag, can check this link [penn_treebank_pos](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

## Stemming and Lemmatization
  * Basic idea: reduce the words into base form. for example, "dog" and "dogs" could be same, same for "jump" and "jumping".
  * stemming is more basic or "crude" version
### Stemmer
NLTK has several different stemmer. 
```python
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
porter_stemmer.stem("wolves")
#Output: 'wolv'
```
###lemmatizer
```python
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
lemma.lemmatize("wolves")
#Output:'wolf'
```

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

##NER: Named entity recognization 
Entity example:
  * Albert Einstein: Name
  * Google: orgnization

```python
s = "albert Einstein was born on March 14, 1879"
tags = nltk.pos_tag(s)
tags:
[('a', 'DT'),
 ('l', 'NN'),
 ('b', 'NN'),
 ('e', 'NN'),
 ('r', 'NN'),
 ('t', 'NN'),
 (' ', 'NNP'),
 ('E', 'NNP'),
 ('i', 'NN'),
 ('n', 'VBP'),
 ('s', 'NN'),
 ('t', 'NN'),
 ('e', 'NN'),
 ('i', 'NN'),
 ('n', 'VBP'),
 (' ', 'NNP'),
 ('w', 'VBP'),
 ('a', 'DT'),
 ('s', 'NN'),
 (' ', 'NN'),
 ('b', 'NN'),
 ('o', 'NN'),
 ('r', 'NN'),
 ('n', 'JJ'),
 (' ', 'NNP'),
 ('o', 'NN'),
 ('n', 'NN'),
 (' ', 'NNP'),
 ('M', 'NNP'),
 ('a', 'DT'),
 ('r', 'NN'),
 ('c', 'NN'),
 ('h', 'NN'),
 (' ', 'VBD'),
 ('1', 'CD'),
 ('4', 'CD'),
 (',', ','),
 (' ', 'VBD'),
 ('1', 'CD'),
 ('8', 'CD'),
 ('7', 'CD'),
 ('9', 'CD')]
```

we do not want this, so split the string first
```python
s = "albert Einstein was born on March 14, 1879".split()
tags = nltk.pos_tag(s)
tags:
[('albert', 'JJ'),
 ('Einstein', 'NNP'),
 ('was', 'VBD'),
 ('born', 'VBN'),
 ('on', 'IN'),
 ('March', 'NNP'),
 ('14,', 'CD'),
 ('1879', 'CD')]
```
Now we want to use *NER* to chunk
```python
#install nltk package 
nltk.ne_chunk(tags)
# get the parse tree
Tree('S', [('albert', 'JJ'), Tree('PERSON', [('Einstein', 'NNP')]), ('was', 'VBD'), ('born', 'VBN'), ('on', 'IN'), ('March', 'NNP'), ('14,', 'CD'), ('1879', 'CD')])
# Or we can virtualize it
nltk.ne_chunk(tags).draw()
```

Another example
```python
s = "steve jobs was Apple CEO"
tags = nltk.pos_tag(s.split())
nltk.ne_chunk(tags)
```

# Latent Semantic Analysis(LSA)
  * synonym: 
    * Used for multiple words have same meaning. 
    Example:"buy" and "purchase", "big" and "large" "quick" and "speed"
  * Polysemy
    * One word has multiple meaning
    * "Man", "Milk"

we can assigne latent variables such as 
  * z = 0.7* computer + 0.5 * PC + 0.3*laptop
  
## underline Math for LSA
  * LSA is just **SVD** on the term document matrix
  * SVD simple version will be PCA
### PCA
  * decorrelate the data
  * transformed data is ordered by information latent
  * Dimension reduction
  
  > removing information != decreasing predictive probablity
  
  > denosing/ smooth / improving generilization 

### SVD

SVD just dose both PCAs on same time on **X^TX** and **XX^T**
A great tutorial on SVD/word2vec are in [stanford cs224n notes](http://web.stanford.edu/class/cs224n/lecture_notes/cs224n-2017-notes1.pdf)

## Example of LSA on Book Title 
some process we need to do: 

like book titles have something 3rd edition
```python
tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
```
#Word2vec

referring to [Word2vec Tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

## The Skip-Gram Model
Goal: We’re going to train the neural network to do the following. Given a specific word in the middle of a sentence (the input word), 
look at the words nearby and pick one at random. The network is going to tell us the probability for every word in 
our vocabulary of being the “nearby word” that we chose.
 
We’ll train the neural network to do this by feeding it word pairs found in our training documents. 
The below example shows some of the training samples (word pairs) we would take from the sentence 
“The quick brown fox jumps over the lazy dog.” I’ve used a small window size of 2 just for the example. 
The word highlighted in blue is the input word.
![alt text][word_pair]
[word_pair]: https://github.com/zhangruiskyline/NLP_demo/blob/master/img/training_data.png "word pair train"

If two different words have very similar “contexts” (that is, what words are likely to appear around them), 
then our model needs to output very similar results for these two words. And one way for the network to output similar 
context predictions for these two words is if the word vectors are similar. So, if two words have similar contexts, 
then our network is motivated to learn similar word vectors for these two words! Ta da!

And what does it mean for two words to have similar contexts? I think you could expect that synonyms 
like “intelligent” and “smart” would have very similar contexts. Or that words that are related, 
like “engine” and “transmission”, would probably have similar contexts as well.

##Model Details
let’s say we have a vocabulary of 10,000 unique words.
We’re going to represent an input word like “ants” as a one-hot vector. This vector will have 10,000 components (one for every word in our vocabulary) 
and we’ll place a “1” in the position corresponding to the word “ants”, and 0s in all of the other positions.

The output of the network is a single vector (also with 10,000 components) containing, for every word in our vocabulary, 
the probability that a randomly selected nearby word is that vocabulary word.
![alt text][NN_structure]
[NN_structure]: https://github.com/zhangruiskyline/NLP_demo/blob/master/img/skip_gram_net_arch.png "Network structure"

When training this network on word pairs, the input is a one-hot vector representing the input word and 
the training output is also a one-hot vector representing the output word. 
But when you evaluate the trained network on an input word, the output vector will actually be a probability distribution 
(i.e., a bunch of floating point values, not a one-hot vector).

## Hidden layer
For our example, we’re going to say that we’re learning word vectors with 300 features. 
So the hidden layer is going to be represented by a weight matrix with 10,000 rows 
(one for every word in our vocabulary) and 300 columns (one for every hidden neuron).

If you look at the rows of this weight matrix, these are actually what will be our *word vectors*!
> So the end goal of all of this is really just to learn this hidden layer weight matrix – 
the output layer we’ll just toss when we’re done!

we can take a look at this in another way:
 If you multiply a 1 x 10,000 one-hot vector by a 10,000 x 300 matrix, 
 it will effectively just select the matrix row corresponding to the “1”. Here’s a small example to give you a visual.
 ![alt text][one_hot_example]
[one_hot_example]: https://github.com/zhangruiskyline/NLP_demo/blob/master/img/word2vec_weight_matrix_lookup_table.png "One hot vector x hidden word2vec"

This means that the hidden layer of this model is really just operating as a lookup table. 
The output of the hidden layer is just the “word vector” for the input word.
![alt text][matrix]
[matrix]: https://github.com/zhangruiskyline/NLP_demo/blob/master/img/matrix_mult_w_one_hot.png "example to use word2vec"

##Output
The __*1 x 300*__ word vector for “ants” then gets fed to the output layer. 
The output layer is a softmax regression classifier. 
but the gist of it is that each output neuron (one per word in our vocabulary!) 
will produce an output between 0 and 1, and the sum of all these output values will add up to 1.
Specifically, each output neuron has a weight vector which it multiplies against the word vector from the hidden layer, 
then it applies the function __*exp(x)*__ to the result. Finally, in order to get the outputs to sum up to 1, 
we divide this result by the sum of the results from all 10,000 output nodes.

![alt text][out_weight]
[out_weight]: https://github.com/zhangruiskyline/NLP_demo/blob/master/img/output_weights_function.png

##Intuition

If two different words have very similar “contexts” (that is, what words are likely to appear around them), then our model needs to output very similar results for these two words. 
And one way for the network to output similar context predictions for these two words is if the word vectors are similar.
 So, if two words have similar contexts, then our network is motivated to learn similar word vectors for these two words! Ta da!

And what does it mean for two words to have similar contexts? I think you could expect that synonyms like “intelligent” and “smart” would have very similar contexts. Or that words that are related, 
like “engine” and “transmission”, would probably have similar contexts as well.
Here’s an illustration of calculating the output of the output neuron for the word “car”.


# Deep Learning NLP

## Dataset
Dataset: [wiki dump data](https://dumps.wikimedia.org)

Convert from XML to txt:
[wiki xml to txt](https://github.com/yohasebe/wp2txt)
```shell
$ gem install wp2txt
wp2txt -i <filename>
```
### Pos Tag dataset
For Pos Tag, use the dataset: 
[Chunk dataset](http://www.cnts.ua.ac.be/conll2000/chunking/)

### NER dataset
[twitter ner](https://github.com/aritter/twitter_nlp/tree/master/data/annotated)

###sentiment analysis
[stanford NLP](http://nlp.stanford.edu/sentiment/)


# Python 3 change

> use dict.items() instead of dict.iteritems()

> use range() instead of xrange()



