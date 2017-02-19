# NLP_demo

# Start Trial 
  1. [Parts-of-speech.Info](http://parts-of-speech.info)

  2. [Stanford parser](http://nlp.stanford.edu:8080/parser/index.jsp)

# Project 1: Spam detector

[spam data set](https://archive.ics.uci.edu/ml/datasets/Spambase)

# project 2: sentiment analysis
[sentiment data](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)

# Section 1 : NLTK

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

# Section 2: Latent Semantic Analysis(LSA)
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

# Section 3: Word embedding 
 > dimension reduction(like PCA)
 > Finding a high level representation of data
 > decorrelate: if 99% docuements that contains "car" also contains "vehicle", then we do not need two dimensions to represent
 > Words = categorical variables -> One hot encoding 
 > 1 million words -> 1 millision dimension vector to represent
 
*Problem of one hot encoding?*
 
 > dimension too high
 > (car, vehicle) are more similar with (car, ocean)
 > but one hot encoding has all words in same distance apart
 > [1,0,0] - [0,1,0] = 2(Manhatan distance)
 > do not know two words are similar 
 
*Word embedding*

 > vocabuary in row, documents in col
 > which document a certain word shows up is *__feature__*
 > example: if certain words always/only shows in financial report, only need one dimension to represent
 > Unsupervised learning: "financial report" is hidden cause/latent variable. The actually document is generated from a distribution
 which describe what they looks like 
 > Word embedding has meaning : King - Man = Queue - woman
 
*PCA way to understand*

```python
from sklearn.decomposition import PCA
X = doc_matrix
model = PCA()
Z = model.fit_transform(X)
# X is VxN, and Z is VxD, D<<N
```
In this case, Z is word embedding, each row has D dimensional 

## word analogies
 > king - man = queue - woman ?

```
vec(king) = We[word2idx["king"]]
v0 = vec(king) - vec(man) + vec(woman)
# find one that closes to v0, return that word
``` 
 > distance in word distance:  
 > Cos distance: dist(a,b) = 1 - cos(a,b)
 
# TF-IDF and t-SNE
example: consider each paragraph a document(not sentence otherwise training time is too long)


# Section 4: Word2vec

* Referring to [Word2vec Tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
* Another useful linke is [Tensorflow Word2vec](https://www.tensorflow.org/tutorials/word2vec)

## Main idea:

* predict every word and its context words
* Two main algorithms
> CBOW(Continuous Bag of Words): predict target word from a bag of words. CBOW is trained to predict the target word t from the contextual words that surround it
> Skip Gram: predict target words from bag of words(position independent). The direction of the prediction is simply inverted,

* Two efficient training methods
> Hierarchical softmax
> Negative sampling
 


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

## Wordnet example
 > taxonomy like WordNet: hypernyms (is-a) relationships

```python
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('text8')
#  text8 from http://mattmahoney.net/dc/text8.zip is wikipedia data: http://mattmahoney.net/dc/textdata.html see "Relationship of Wikipedia Text to Clean Text" 
model = word2vec.Word2Vec(sentences, size=200)
## will train the data
2017-02-17 22:03:53,854 : INFO : collecting all words and their counts
2017-02-17 22:03:53,860 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
2017-02-17 22:04:00,626 : INFO : collected 253854 word types from a corpus of 17005207 raw words and 1701 sentences
2017-02-17 22:04:00,626 : INFO : Loading a fresh vocabulary
2017-02-17 22:04:00,884 : INFO : min_count=5 retains 71290 unique words (28% of original 253854, drops 182564)
2017-02-17 22:04:00,884 : INFO : min_count=5 leaves 16718844 word corpus (98% of original 17005207, drops 286363)
2017-02-17 22:04:01,114 : INFO : deleting the raw counts dictionary of 253854 items
2017-02-17 22:04:01,137 : INFO : sample=0.001 downsamples 38 most-common words
2017-02-17 22:04:01,137 : INFO : downsampling leaves estimated 12506280 word corpus (74.8% of prior 16718844)
2017-02-17 22:04:01,137 : INFO : estimated required memory for 71290 words and 200 dimensions: 149709000 bytes
2017-02-17 22:04:01,419 : INFO : resetting layer weights
2017-02-17 22:04:02,553 : INFO : training model with 3 workers on 71290 vocabulary and 200 features, using sg=0 hs=0 sample=0.001 negative=5 window=5
2017-02-17 22:04:02,553 : INFO : expecting 1701 sentences, matching count from corpus used for vocabulary survey
2017-02-17 22:04:03,573 : INFO : PROGRESS: at 0.78% examples, 479620 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:04:04,590 : INFO : PROGRESS: at 1.69% examples, 517727 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:04:05,590 : INFO : PROGRESS: at 2.70% examples, 554621 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:06,593 : INFO : PROGRESS: at 3.57% examples, 551485 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:07,597 : INFO : PROGRESS: at 4.37% examples, 540852 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:08,601 : INFO : PROGRESS: at 5.34% examples, 551815 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:09,605 : INFO : PROGRESS: at 6.24% examples, 554666 words/s, in_qsize 4, out_qsize 1
2017-02-17 22:04:10,616 : INFO : PROGRESS: at 7.22% examples, 561685 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:04:11,626 : INFO : PROGRESS: at 8.07% examples, 557540 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:04:12,628 : INFO : PROGRESS: at 8.90% examples, 554115 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:13,634 : INFO : PROGRESS: at 9.84% examples, 557206 words/s, in_qsize 4, out_qsize 1
2017-02-17 22:04:14,646 : INFO : PROGRESS: at 10.83% examples, 561928 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:04:15,658 : INFO : PROGRESS: at 11.78% examples, 564115 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:16,665 : INFO : PROGRESS: at 12.78% examples, 568347 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:17,665 : INFO : PROGRESS: at 13.76% examples, 571182 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:04:18,688 : INFO : PROGRESS: at 14.74% examples, 573476 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:19,691 : INFO : PROGRESS: at 15.18% examples, 555156 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:20,706 : INFO : PROGRESS: at 15.57% examples, 537083 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:21,707 : INFO : PROGRESS: at 16.06% examples, 525068 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:04:22,720 : INFO : PROGRESS: at 16.55% examples, 513907 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:23,731 : INFO : PROGRESS: at 17.41% examples, 514751 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:04:24,731 : INFO : PROGRESS: at 18.42% examples, 519996 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:04:25,738 : INFO : PROGRESS: at 19.41% examples, 523912 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:26,746 : INFO : PROGRESS: at 20.34% examples, 526045 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:27,767 : INFO : PROGRESS: at 21.21% examples, 526090 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:28,772 : INFO : PROGRESS: at 22.15% examples, 528123 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:04:29,775 : INFO : PROGRESS: at 22.89% examples, 525641 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:30,852 : INFO : PROGRESS: at 23.49% examples, 518901 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:31,874 : INFO : PROGRESS: at 23.92% examples, 509793 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:32,960 : INFO : PROGRESS: at 24.47% examples, 502960 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:33,964 : INFO : PROGRESS: at 24.87% examples, 494989 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:34,981 : INFO : PROGRESS: at 25.30% examples, 487878 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:36,037 : INFO : PROGRESS: at 25.67% examples, 479400 words/s, in_qsize 5, out_qsize 1
2017-02-17 22:04:37,045 : INFO : PROGRESS: at 26.41% examples, 478988 words/s, in_qsize 5, out_qsize 1
2017-02-17 22:04:38,056 : INFO : PROGRESS: at 27.29% examples, 481082 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:39,067 : INFO : PROGRESS: at 28.22% examples, 483660 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:40,069 : INFO : PROGRESS: at 28.92% examples, 482576 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:41,069 : INFO : PROGRESS: at 29.45% examples, 478691 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:42,083 : INFO : PROGRESS: at 30.39% examples, 481336 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:43,085 : INFO : PROGRESS: at 31.39% examples, 484960 words/s, in_qsize 5, out_qsize 1
2017-02-17 22:04:44,086 : INFO : PROGRESS: at 32.28% examples, 486604 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:45,101 : INFO : PROGRESS: at 33.19% examples, 488477 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:46,104 : INFO : PROGRESS: at 34.20% examples, 491820 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:47,144 : INFO : PROGRESS: at 35.09% examples, 492579 words/s, in_qsize 6, out_qsize 1
2017-02-17 22:04:48,160 : INFO : PROGRESS: at 36.01% examples, 494100 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:04:49,161 : INFO : PROGRESS: at 37.00% examples, 496709 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:50,165 : INFO : PROGRESS: at 38.05% examples, 500044 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:04:51,175 : INFO : PROGRESS: at 38.93% examples, 500865 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:04:52,181 : INFO : PROGRESS: at 39.92% examples, 503116 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:53,210 : INFO : PROGRESS: at 40.58% examples, 501018 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:04:54,231 : INFO : PROGRESS: at 41.46% examples, 501596 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:55,239 : INFO : PROGRESS: at 42.46% examples, 503800 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:04:56,239 : INFO : PROGRESS: at 43.35% examples, 504796 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:04:57,246 : INFO : PROGRESS: at 44.28% examples, 506138 words/s, in_qsize 1, out_qsize 0
2017-02-17 22:04:58,259 : INFO : PROGRESS: at 45.26% examples, 507978 words/s, in_qsize 4, out_qsize 1
2017-02-17 22:04:59,270 : INFO : PROGRESS: at 46.27% examples, 510230 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:05:00,274 : INFO : PROGRESS: at 47.33% examples, 512956 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:05:01,274 : INFO : PROGRESS: at 48.37% examples, 515383 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:02,286 : INFO : PROGRESS: at 49.34% examples, 516802 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:03,287 : INFO : PROGRESS: at 50.25% examples, 517756 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:04,288 : INFO : PROGRESS: at 51.01% examples, 517036 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:05,294 : INFO : PROGRESS: at 51.84% examples, 517085 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:06,298 : INFO : PROGRESS: at 52.66% examples, 517055 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:07,312 : INFO : PROGRESS: at 53.54% examples, 517481 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:08,331 : INFO : PROGRESS: at 54.46% examples, 518227 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:09,370 : INFO : PROGRESS: at 55.39% examples, 518657 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:10,403 : INFO : PROGRESS: at 56.28% examples, 518942 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:11,420 : INFO : PROGRESS: at 56.90% examples, 516823 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:12,443 : INFO : PROGRESS: at 57.57% examples, 515235 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:13,449 : INFO : PROGRESS: at 58.01% examples, 511915 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:05:14,458 : INFO : PROGRESS: at 58.81% examples, 511602 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:15,472 : INFO : PROGRESS: at 59.59% examples, 511100 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:16,497 : INFO : PROGRESS: at 60.28% examples, 509881 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:17,514 : INFO : PROGRESS: at 61.09% examples, 509698 words/s, in_qsize 4, out_qsize 1
2017-02-17 22:05:18,515 : INFO : PROGRESS: at 61.80% examples, 508680 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:19,518 : INFO : PROGRESS: at 62.53% examples, 507946 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:20,534 : INFO : PROGRESS: at 63.42% examples, 508493 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:21,536 : INFO : PROGRESS: at 64.14% examples, 507716 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:22,552 : INFO : PROGRESS: at 65.04% examples, 508393 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:23,567 : INFO : PROGRESS: at 66.07% examples, 510042 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:05:24,568 : INFO : PROGRESS: at 67.08% examples, 511620 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:25,577 : INFO : PROGRESS: at 68.09% examples, 513033 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:26,590 : INFO : PROGRESS: at 69.02% examples, 513788 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:05:27,607 : INFO : PROGRESS: at 69.83% examples, 513653 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:28,609 : INFO : PROGRESS: at 70.79% examples, 514702 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:29,617 : INFO : PROGRESS: at 71.46% examples, 513569 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:30,651 : INFO : PROGRESS: at 72.23% examples, 512992 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:31,652 : INFO : PROGRESS: at 72.91% examples, 512019 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:05:32,664 : INFO : PROGRESS: at 73.62% examples, 511173 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:33,673 : INFO : PROGRESS: at 74.39% examples, 510855 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:34,696 : INFO : PROGRESS: at 75.23% examples, 510762 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:35,704 : INFO : PROGRESS: at 75.97% examples, 510137 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:36,710 : INFO : PROGRESS: at 76.47% examples, 508020 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:37,773 : INFO : PROGRESS: at 77.25% examples, 507447 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:38,806 : INFO : PROGRESS: at 77.58% examples, 504128 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:39,814 : INFO : PROGRESS: at 78.30% examples, 503512 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:40,820 : INFO : PROGRESS: at 79.25% examples, 504400 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:41,837 : INFO : PROGRESS: at 80.24% examples, 505417 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:42,898 : INFO : PROGRESS: at 81.11% examples, 505453 words/s, in_qsize 4, out_qsize 1
2017-02-17 22:05:43,910 : INFO : PROGRESS: at 81.67% examples, 503828 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:44,913 : INFO : PROGRESS: at 82.75% examples, 505476 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:05:45,934 : INFO : PROGRESS: at 83.73% examples, 506378 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:46,936 : INFO : PROGRESS: at 84.64% examples, 507036 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:47,955 : INFO : PROGRESS: at 85.77% examples, 508899 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:48,957 : INFO : PROGRESS: at 86.81% examples, 510267 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:50,001 : INFO : PROGRESS: at 87.76% examples, 510881 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:51,023 : INFO : PROGRESS: at 88.52% examples, 510497 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:52,034 : INFO : PROGRESS: at 89.63% examples, 512123 words/s, in_qsize 5, out_qsize 1
2017-02-17 22:05:53,034 : INFO : PROGRESS: at 90.26% examples, 511085 words/s, in_qsize 4, out_qsize 1
2017-02-17 22:05:54,039 : INFO : PROGRESS: at 91.35% examples, 512578 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:55,060 : INFO : PROGRESS: at 92.04% examples, 511801 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:05:56,061 : INFO : PROGRESS: at 92.33% examples, 508912 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:05:57,076 : INFO : PROGRESS: at 92.84% examples, 507166 words/s, in_qsize 4, out_qsize 1
2017-02-17 22:05:58,145 : INFO : PROGRESS: at 93.35% examples, 505209 words/s, in_qsize 6, out_qsize 0
2017-02-17 22:05:59,152 : INFO : PROGRESS: at 93.76% examples, 503067 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:06:00,166 : INFO : PROGRESS: at 94.87% examples, 504699 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:06:01,185 : INFO : PROGRESS: at 95.98% examples, 506035 words/s, in_qsize 3, out_qsize 0
2017-02-17 22:06:02,188 : INFO : PROGRESS: at 97.07% examples, 507494 words/s, in_qsize 3, out_qsize 0
2017-02-17 22:06:03,192 : INFO : PROGRESS: at 98.18% examples, 509003 words/s, in_qsize 5, out_qsize 0
2017-02-17 22:06:04,200 : INFO : PROGRESS: at 99.33% examples, 510669 words/s, in_qsize 4, out_qsize 0
2017-02-17 22:06:04,778 : INFO : worker thread finished; awaiting finish of 2 more threads
2017-02-17 22:06:04,786 : INFO : worker thread finished; awaiting finish of 1 more threads
2017-02-17 22:06:04,788 : INFO : worker thread finished; awaiting finish of 0 more threads
2017-02-17 22:06:04,788 : INFO : training on 85026035 raw words (62530504 effective words) took 122.2s, 511606 effective words/s


## some test example:
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=2) 
model.most_similar(['man'])

## we can also save model
model.save('text8.model')
model.save_word2vec_format('text.model.bin', binary=True)
```

##Glove

```python
import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove
sentences = list(itertools.islice(Text8Corpus('text8'),None))
corpus = Corpus()
corpus.fit(sentences, window=10)
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
```

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



