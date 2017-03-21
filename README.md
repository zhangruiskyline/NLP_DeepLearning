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
* Presentation[NIPS word2vec presentation](https://docs.google.com/file/d/0B7XkCwpI5KDYRWRnd1RzWXQ2TWc/edit)
* A complete resource on word2vec[resource on word2vec](http://mccormickml.com/2016/04/27/word2vec-resources/#alex-minnaars-tutorials)


## Main idea:

*Predict every word and its context words*, You can get a lot of value by representing a word by means of its neighbors
“You shall know a word by the company it keeps”. One of most successful idea in modern NLP
###Two main algorithms
* CBOW(Continuous Bag of Words): predict target word from a bag of words. CBOW is trained to predict the target word t from the contextual words that surround it
* Skip Gram: predict target words from bag of words(position independent). The direction of the prediction is simply inverted, predicting the context given a word

![alt text](https://github.com/zhangruiskyline/NLP_demo/blob/master/img/CBOW_and_Skip_gram.png)

> * CBOW: Fast to train than the skip-gram, slightly better accuracy for the frequent words
CBOW is good at syntatic learning 
> * Skip-gram: Slow to train, represents well even rare words or phrases. skip gram is good at semantic learning


This can get even a bit more complicated if you consider that there are two different ways how to train the models: the normalized hierarchical softmax, and the un-normalized negative sampling. Both work quite differently.
In CBOW the vectors from the context words are averaged before predicting the center word. In skip-gram there is no averaging of embedding vectors

Let's take a look at these two methods in more detail

* CBOW:
 
> By the Distributional Hypothesis (Firth, 1957; see also the Wikipedia page on Distributional semantics), words with similar distributional properties (i.e. that co-occur regularly) tend to share some aspect of semantic meaning. For example, we may find several sentences in the training set such as "citizens of X protested today" where X (the target word t) may be names of cities or countries that are semantically related. 
the goal is to maximize P(t | c) over the training set.  I am simplifying somewhat, but you can show that this probability is roughly inversely proportional to the distance between the current vectors assigned to t and to c. Since this model is trained in an online setting (one example at a time), at time T the goal is therefore to take a small step (mediated by the "learning rate") in order to minimize the distance between the current vectors for t and c (and thereby increase the probability P(t |c)).  By repeating this process over the entire training set, we have that vectors for words that habitually co-occur tend to be nudged closer together, and by gradually lowering the learning rate, this process converges towards some final state of the vectors. 

* Skip Gram

> For the skipgram, the direction of the prediction is simply inverted, i.e. now we try to predict P(citizens | X), P(of | X), etc. This turns out to learn finer-grained vectors when one trains over more data. The main reason is that the CBOW smooths over a lot of the distributional statistics by averaging over all context words while the skipgram does not. With little data, this "regularizing" effect of the CBOW turns out to be helpful, but since data is the ultimate regularizer the skipgram is able to extract more information when more data is available.

### Two efficient training methods
* Hierarchical softmax


* Negative sampling

Traditionally, each training sample will tweak all of the weights in the neural network.
In the example I gave, we had word vectors with 300 components, and a vocabulary of 10,000 words. Recall that the neural network had two weight matrices–a hidden layer and output layer. 
Both of these layers would have a weight matrix with 300 x 10,000 = 3 million weights each!

 > Negative sampling addresses this by having each training sample only modify a small percentage of the weights, rather than all of them. 

## Continuous Bag of Words (CBOW)
The mode of CBOW can be shown as following:

![alt text](https://github.com/zhangruiskyline/NLP_demo/blob/master/img/CBOW.png)

we can see that for *C* context words, there will be *C* input vectors and output will be the predicting word, which is calculated 
by the average of all input vectors


## The Skip-Gram Model

### Skip gram in example: word pairs
Goal: We’re going to train the neural network to do the following. Given a specific word in the middle of a sentence (the input word), 
look at the words nearby and pick one at random. The network is going to tell us the probability for every word in 
our vocabulary of being the “nearby word” that we chose.
 
We’ll train the neural network to do this by feeding it word pairs found in our training documents. 
The below example shows some of the training samples (word pairs) we would take from the sentence 
“The quick brown fox jumps over the lazy dog.” I’ve used a small window size of 2 just for the example. 
The word highlighted in blue is the input word.

![alt text](https://github.com/zhangruiskyline/NLP_demo/blob/master/img/training_data.png) "word pair train"

If two different words have very similar “contexts” (that is, what words are likely to appear around them), 
then our model needs to output very similar results for these two words. And one way for the network to output similar 
context predictions for these two words is if the word vectors are similar. So, if two words have similar contexts, 
then our network is motivated to learn similar word vectors for these two words! Ta da!

And what does it mean for two words to have similar contexts? I think you could expect that synonyms 
like “intelligent” and “smart” would have very similar contexts. Or that words that are related, 
like “engine” and “transmission”, would probably have similar contexts as well.

####Model Details
let’s say we have a vocabulary of 10,000 unique words.
We’re going to represent an input word like “ants” as a one-hot vector. This vector will have 10,000 components (one for every word in our vocabulary) 
and we’ll place a “1” in the position corresponding to the word “ants”, and 0s in all of the other positions.

The output of the network is a single vector (also with 10,000 components) containing, for every word in our vocabulary, 
the probability that a randomly selected nearby word is that vocabulary word.
![alt text](https://github.com/zhangruiskyline/NLP_demo/blob/master/img/skip_gram_net_arch.png) 

When training this network on word pairs, the input is a one-hot vector representing the input word and 
the training output is also a one-hot vector representing the output word. 
But when you evaluate the trained network on an input word, the output vector will actually be a probability distribution 
(i.e., a bunch of floating point values, not a one-hot vector).

#### Hidden layer
For our example, we’re going to say that we’re learning word vectors with 300 features. 
So the hidden layer is going to be represented by a weight matrix with 10,000 rows 
(one for every word in our vocabulary) and 300 columns (one for every hidden neuron).

If you look at the rows of this weight matrix, these are actually what will be our *word vectors*!
> So the end goal of all of this is really just to learn this hidden layer weight matrix – 
the output layer we’ll just toss when we’re done!

we can take a look at this in another way:
 If you multiply a 1 x 10,000 one-hot vector by a 10,000 x 300 matrix, 
 it will effectively just select the matrix row corresponding to the “1”. Here’s a small example to give you a visual.
 ![alt text](https://github.com/zhangruiskyline/NLP_demo/blob/master/img/word2vec_weight_matrix_lookup_table.png)

This means that the hidden layer of this model is really just operating as a lookup table. 
The output of the hidden layer is just the “word vector” for the input word.
![alt text][matrix]
[matrix]: https://github.com/zhangruiskyline/NLP_demo/blob/master/img/matrix_mult_w_one_hot.png "example to use word2vec"

####Output
The __*1 x 300*__ word vector for “ants” then gets fed to the output layer. 
The output layer is a softmax regression classifier. 
but the gist of it is that each output neuron (one per word in our vocabulary!) 
will produce an output between 0 and 1, and the sum of all these output values will add up to 1.
Specifically, each output neuron has a weight vector which it multiplies against the word vector from the hidden layer, 
then it applies the function __*exp(x)*__ to the result. Finally, in order to get the outputs to sum up to 1, 
we divide this result by the sum of the results from all 10,000 output nodes.

![alt text](https://github.com/zhangruiskyline/NLP_demo/blob/master/img/output_weights_function.png)

###skip gram model in real 
In our example, we only use one word to predict its pair word. In real application, we use one word to predict multiple context words.
On the output layer, instead of outputing one multinomial distribution, we are outputing C multinomial distributions. Each output is computed using the same hidden→output i
is the same matrix:

![alt text][skip_gram]
[skip_gram]:https://github.com/zhangruiskyline/NLP_demo/blob/master/img/skip_gram.png

###Intuition

If two different words have very similar “contexts” (that is, what words are likely to appear around them), then our model needs to output very similar results for these two words. 
And one way for the network to output similar context predictions for these two words is if the word vectors are similar.
 So, if two words have similar contexts, then our network is motivated to learn similar word vectors for these two words! Ta da!

And what does it mean for two words to have similar contexts? I think you could expect that synonyms like “intelligent” and “smart” would have very similar contexts. Or that words that are related, 
like “engine” and “transmission”, would probably have similar contexts as well.
Here’s an illustration of calculating the output of the output neuron for the word “car”.

###Negative sampling
> * Besides the positive training data(in skip gram, it is surrounding context), we need to have negative data
to train the network so that unncessary context will be used for punishment. 
> * However, if we tell the network *ALL* not in target words, will be too much,
 A lot of time spent is spent on words not in target too many weights update in each training epoch of GD
> * We sample words from a set of words that are not in context, call these "negative sampling"

A example to show why we need to have negative sampling:

When training the network on the word pair (“fox”, “quick”), recall that the “label” or “correct output” of the network is a one-hot vector. That is, for the output neuron corresponding to “quick” to output a 1, and for all of the other thousands of output neurons to output a 0.
With negative sampling, we are instead going to randomly select just a small number of “negative” words (let’s say 5) to update the weights for. (In this context, a “negative” word is one for which we want the network to output a 0 for). We will also still update the weights for our “positive” word (which is the word “quick” in our current example).

The paper says that selecting 5-20 words works well for smaller datasets, and you can get away with only 2-5 words for large datasets.

> How to choose negavtive sampling:
* Distribution of word frequency: *P(W) = count(W)/total* 
* Word probablity has a long tail 
* Use *P(W)^0.75* to raise probablity of less happened words
* If a word apprears a lot, but not in context, we want to give away

### Cost function with negative sampling
The Cost function will include both the context and negative sampling
![alt text][cost_func]
[cost_func]:https://github.com/zhangruiskyline/NLP_demo/blob/master/img/skip_gram_cost_func.png

## Word2vec Application
 > Word2vec is mostly used for embedding, not for classification

Eg: we have "I love dog and cat" and "My dog ate my food", so if we use dog to predict two sentence, doe not make much sense

### How to train

> **gensim** is the most widly used lib

Default gensim dose not have negative sampling enabled

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

#Glove

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

# Section 2: NLP Common Pre-Process Techs 

## Documentation Matrix
A document-term matrix or term-document matrix is a mathematical matrix that describes the frequency of terms that occur in a collection of documents.

In a document-term matrix, rows correspond to documents in the collection and columns correspond to terms. There are various schemes for determining the value that each entry in the matrix should take. One such scheme is tf-idf. They are useful in the field of natural language processing.

## General Concept:
When creating a database of terms that appear in a set of documents the document-term matrix contains rows corresponding to the documents and columns corresponding to the terms. For instance if one has the following two (short) documents:

D1 = "I like databases"
D2 = "I hate databases",
then the document-term matrix would be:

Doc-Term | I | Like | hate | databse
---| --- | --- | --- | ---
D1	|1 |	1|	0|	1
D2 | 1	|0	|1	|1

> Python sklearn Lib [Link](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
```python
from sklearn.feature_extraction.text import CountVectorizer
```

which shows which documents contain which terms and how many times they appear.

## TF-IDF

### Term Frenquency 
In information retrieval or text mining, the term frequency – inverse document frequency (also called tf-idf), is a well know method to evaluate how important is a word in a document. tf-idf are is a very interesting way to convert the textual representation of information into a Vector Space Model (VSM), or into sparse features, we’ll discuss more about it later, but first, let’s try to understand what is tf-idf and the VSM.

 > Going to the vector space
 
The first step in modeling the document into a vector space is to create a dictionary of terms present in documents. To do that, you can simple select all terms from the document and convert it to a dimension in the vector space, but we know that there are some kind of words (stop words) that are present in almost all documents, and what we’re doing is extracting important features from documents, features do identify them among other similar documents, so using terms like “the, is, at, on”, etc.. isn’t going to help us, so in the information extraction, we’ll just ignore them.
Let’s take the documents below to define our (stupid) document space:
```
Train Document Set:
d1: The sky is blue.
d2: The sun is bright.
Test Document Set:
d3: The sun in the sky is bright.
d4: We can see the shining sun, the bright sun.
```

Now, what we have to do is to create a __*index vocabulary*__ (dictionary) of the words of the train document set, using the documents d1 and d2 from the document set, we’ll have the following index vocabulary denoted as __*E(t)*__ where the t is the term:

![alt text][tf_features.png]
[tf_features.png]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/tf_features.png


Note that the terms like “is” and “the” were ignored as cited before. Now that we have an index vocabulary, we can convert the test document set into a vector space where each term of the vector is indexed as our index vocabulary, so the first term of the vector represents the “blue” term of our vocabulary, the second represents “sun” and so on. Now, we’re going to use the term-frequency to represent each term in our vector space; the term-frequency is nothing more than a measure of how many times the terms present in our vocabulary __*E(t)*__ are present in the documents d3 or d4, we define the term-frequency as a couting function:


Tf is defined as:

![alt text][tf_define]
[tf_define]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/tf_1.png

where you have:

![alt text][tf_define_2]
[tf_define_2]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/tf_2.png

TF matrix

TF_Matrix | blue | sun | bright | sky
---| --- | --- | --- | ---
d3	|0 |	1|	1|	1
d4 | 0	|2	|1	|0

As you may have noted, these matrices representing the term frequencies tend to be very sparse (with majority of terms zeroed), and that’s why you’ll see a common representation of these matrix as sparse matrices.


### The inverse document frequency (IDF) weight
In the first post, we learned how to use the term-frequency to represent textual information in the vector space. However, the main problem with the term-frequency approach is that it scales up frequent terms and scales down rare terms which are empirically more informative than the high frequency terms. 

The basic intuition is that a term that occurs frequently in many documents is not a good discriminator, and really makes sense (at least in many experimental tests); the important question here is: why would you, in a classification problem for instance, emphasize a term which is almost present in the entire corpus of your documents ?

The tf-idf weight comes to solve this problem. What tf-idf gives is how important is a word to a document in a collection, and that’s why tf-idf incorporates local and global parameters, because it takes in consideration not only the isolated term but also the term within the document collection. What tf-idf then does to solve that problem, is to scale down the frequent terms while scaling up the rare terms; a term that occurs 10 times more than another isn’t 10 times more important than it, that’s why tf-idf uses the logarithmic scale to do that.

But let’s go back to our definition of the __tf(t,d)__ which is actually the term count of the term t in the document d. The use of this simple term frequency could lead us to problems like keyword spamming, which is when we have a repeated term in a document with the purpose of improving its ranking on an IR (Information Retrieval) system or even create a bias towards long documents, making them look more important than they are just because of the high frequency of the term in the document.

To overcome this problem, the term frequency __tf(t,d)__ of a document on a vector space is usually also normalized. 

 > idf (inverse document frequency) is then defined:

![alt text][idf]
[idf]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/idf.png

where \left|\{d : t \in d\}\right| is the __number of documents__ where the term t appears, when the term-frequency function satisfies \mathrm{tf}(t,d) \neq 0, we’re only adding 1 into the formula to avoid zero-division.

 > The formula for the tf-idf is then:

![alt text][tfidf]
[tfidf]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/tf_idf.png

and this formula has an important consequence: a high weight of the tf-idf calculation is reached when you have a high term frequency (tf) in the given document (local parameter) and a low document frequency of the term in the whole collection (global parameter).

 > Example 

Your document space can be defined then as D = \{ d_1, d_2, \ldots, d_n \} where n is the number of documents in your corpus, and in our case as D_{train} = \{d_1, d_2\} and D_{test} = \{d_3, d_4\}. The cardinality of our document space is defined by \left|{D_{train}}\right| = 2 and \left|{D_{test}}\right| = 2, since we have only 2 two documents for training and testing, but they obviously don’t need to have the same cardinality.

Now let’s calculate the idf for each feature present in the feature matrix with the term frequency we have calculated:
Since we have 4 features


![alt text][idf_ex1]
[idf_ex1]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/idf_ex1.png

![alt text][idf_ex2]
[idf_ex2]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/idf_ex2.png

![alt text][idf_ex3]
[idf_ex3]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/idf_ex3.png

![alt text][idf_ex4]
[idf_ex4]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/idf_ex4.png

 > The matrix format of idf will be

Now that we have our matrix with the term frequency (M_{train}) and the vector representing the idf for each feature of our matrix (\vec{idf_{train}}), we can calculate our tf-idf weights. What we have to do is a simple multiplication of each column of the matrix M_{train} with the respective \vec{idf_{train}} vector dimension. To do that, we can create a square diagonal matrix called M_{idf} with both the vertical and horizontal dimensions equal to the vector \vec{idf_{train}} dimension:

![alt text][idf_matrix]
[idf_matrix]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/idf_matrix.png

and then multiply it to the term frequency matrix, 

 > so the final result can be defined then as:

![alt text][tfidf_matrix]
[tfidf_matrix]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/tfidf_matrix.png

 > The results looks like

![alt text][tfidf_matrix_val]
[tfidf_matrix_val]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/tfidf_matrix_val.png


![alt text][tfidf_matrix_val_2]
[tfidf_matrix_val_2]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/tfidf_matrix_val_2.png


And finally, we can apply our L2 normalization process to the *TF_IDF* matrix. Please note that this normalization is __row-wise__ because we’re going to handle each row of the matrix as a separated vector to be normalized, and not the matrix as a whole:

![alt text][tfidf_norm]
[tfidf_norm]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/tfidf_norm.png

### Python Example

The first step is to create our training and testing document set and computing the term frequency matrix:

```python
from sklearn.feature_extraction.text import CountVectorizer
train_set = ("The sky is blue.", "The sun is bright.")
test_set = ("The sun in the sky is bright.",
"We can see the shining sun, the bright sun.")
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(train_set)
print "Vocabulary:", count_vectorizer.vocabulary
# Vocabulary: {'blue': 0, 'sun': 1, 'bright': 2, 'sky': 3}
freq_term_matrix = count_vectorizer.transform(test_set)
print freq_term_matrix.todense()
#[[0 1 1 1]
#[0 2 1 0]]
```

Now that we have the frequency term matrix (called freq_term_matrix), we can instantiate the TfidfTransformer, which is going to be responsible to calculate the tf-idf weights for our term frequency matrix:

```python
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
print "IDF:", tfidf.idf_
# IDF: [ 0.69314718 -0.40546511 -0.40546511  0.        ]
```

Note that I’ve specified the norm as L2, this is optional (actually the default is L2-norm), but I’ve added the parameter to make it explicit to you that it it’s going to use the L2-norm. Also note that you can see the calculated idf weight by accessing the internal attribute called idf_. Now that fit() method has calculated the idf for the matrix, let’s transform the freq_term_matrix to the tf-idf weight matrix:

```python
tf_idf_matrix = tfidf.transform(freq_term_matrix)
print tf_idf_matrix.todense()
# [[ 0.         -0.70710678 -0.70710678  0.        ]
# [ 0.         -0.89442719 -0.4472136   0.        ]]
```



# Section 3 : Naive Bayes

Here’s a situation you’ve got into:

You are working on a classification problem and you have generated your set of hypothesis, created features and discussed the importance of variables. Within an hour, stakeholders want to see the first cut of the model.

What will you do? You have hunderds of thousands of data points and quite a few variables in your training data set. In such situation, if I were at your place, I would have used *‘Naive Bayes‘*, which can be extremely fast relative to other classification algorithms. It works on Bayes theorem of probability to predict the class of unknown data set.

## What is Naive Bayes algorithm?
It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is __*unrelated*__ to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.

Bayes theorem provides a way of calculating posterior probability P(c|x) from P(c), P(x) and P(x|c). Look at the equation below:
![alt text][Bayes_theory]
[Bayes_theory]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/Bayes_rule.png

 > * P(c|x) is the posterior probability of class (c, target) given predictor (x, attributes).
 > * P(c) is the prior probability of class.
 > * P(x|c) is the likelihood which is the probability of predictor given class.
 > * P(x) is the prior probability of predictor.

## How Naive Bayes algorithm works?
Let’s understand it using an example. Below I have a training data set of weather and corresponding target variable ‘Play’ (suggesting possibilities of playing). Now, we need to classify whether players will play or not based on weather condition. Let’s follow the below steps to perform it.

 > Step 1: Convert the data set into a frequency table

 > Step 2: Create Likelihood table by finding the probabilities like Overcast probability = 0.29 and probability of playing is 0.64.
 
 > step 3: Step 3: Now, use Naive Bayesian equation to calculate the posterior probability for each class. The class with the highest posterior probability is the outcome of prediction.
 
* Example problem: will we play in sunny?
We can solve it using above discussed method of posterior probability.

![alt text][NB_example]
[NB_example]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/Bayes_example.png

P(Yes | Sunny) = P( Sunny | Yes) * P(Yes) / P (Sunny)

Here we have 

P (Sunny |Yes) = 3/9 = 0.33, P(Sunny) = 5/14 = 0.36, P( Yes)= 9/14 = 0.64

Now, 

P (Yes | Sunny) = 0.33 * 0.64 / 0.36 = 0.60, which has higher probability.

Naive Bayes uses a similar method to predict the probability of different class based on various attributes. This algorithm is mostly used in text classification and with problems having multiple classes.

## Pros and Cons of Naive Bayes?

### Pros

* It is easy and fast to predict class of test data set. It also perform well in multi class prediction
* When assumption of independence holds, a Naive Bayes classifier performs better compare to other models like logistic regression and you need less training data.
* It perform well in case of categorical input variables compared to numerical variable(s). For numerical variable, normal distribution is assumed (bell curve, which is a strong assumption).

### Cons
* If categorical variable has a category (in test data set), which was not observed in training data set, then model will assign a 0 (zero) probability and will be unable to make a prediction. This is often known as “Zero Frequency”. To solve this, we can use the smoothing technique. One of the simplest smoothing techniques is called Laplace estimation.
* On the other side naive Bayes is also known as a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.
* Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.

## Applications

 * Multi class Prediction: 
 
This algorithm is also well known for multi class prediction feature. Here we can predict the probability of multiple classes of target variable.

 * Text classification/ Spam Filtering/ Sentiment Analysis: 
 
Naive Bayes classifiers mostly used in text classification (due to better result in multi class problems and independence rule) have higher success rate as compared to other algorithms. As a result, it is widely used in Spam filtering (identify spam e-mail) and Sentiment Analysis (in social media analysis, to identify positive and negative customer sentiments)

 * Recommendation System: 
 
Naive Bayes Classifier and Collaborative Filtering together builds a Recommendation System that uses machine learning and data mining techniques to filter unseen information and predict whether a user would like a given resource or not

## Naive Bayes in Python and sklearn lib

There are three types of Naive Bayes model under scikit learn library: Referring to [NB in sklearn](http://scikit-learn.org/stable/modules/naive_bayes.html)

* Gaussian: 

It is used in classification and it assumes that features follow a normal distribution.

* Multinomial: 

It is used for discrete counts. For example, let’s say,  we have a text classification problem. Here we can consider bernoulli trials which is one step further and instead of “word occurring in the document”, we have “count how often word occurs in the document”, you can think of it as “number of times outcome number x_i is observed over the n trials”.

* Bernoulli: 

The binomial model is useful if your feature vectors are binary (i.e. zeros and ones). One application would be text classification with ‘bag of words’ model where the 1s & 0s are “word occurs in the document” and “word does not occur in the document” respectively.

 > Here is Python code example
 
```python
#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
X= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])
#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(X, Y)

#Predict Output 
predicted= model.predict([[1,2],[3,4]])
print(predicted)

#Output: ([3,4])
```

Some more examples and detailed information on [Text Classification in NB](http://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note07-2up.pdf)

## Improvement 

Here are some tips for improving power of Naive Bayes Model:

* If continuous features do not have normal distribution,
 
we should use transformation or different methods to convert it in normal distribution.

* If test data set has zero frequency issue, 

apply smoothing techniques “Laplace Correction” to predict the class of test data set.

* Remove correlated features, 

as the highly correlated features are voted twice in the model and it can lead to over inflating importance.

* Naive Bayes classifiers has limited options for parameter tuning
 
like alpha=1 for smoothing, fit_prior=[True|False] to learn class prior probabilities or not and some other options (look at detail here). I would recommend to focus on your  pre-processing of data and the feature selection.

* You might think to apply some classifier combination technique like ensembling, bagging and boosting but these methods would not help. 

Actually, “ensembling, boosting, bagging” won’t help since their purpose is to reduce variance. Naive Bayes has no variance to minimize.

# Section 4: LDA

## Topic model

* A topic model is a type of statistical model for discovering the abstract  that occur in a collection

* Topic models are a suite of algorithms that uncover the  in document collections. These algorithms help us develop new ways to summarize large archives of texts

* Topic models provide a simple way to analyze large volumes of text. A "topic" consists of a  of words that  occur together

## LDA Understanding 

 > LDA can be described as:

![alt text][lda]
[lda]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/lda.png

 > LDA Algorithm in details

Let's go through details:

* initilize the parameter

![alt text][lda_init]
[lda_init]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/lda_init.png

* initilize the topic assignments randomly

![alt text][lda_init_2]
[lda_init_2]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/lda_init_2.png

* iterate

![alt text][lda_iterate]
[lda_iterate]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/lda_iterate.png

* Resample topic for word, given all other words and their current topic assignments

    * Which topics occur in this document?
    * Which topics like the word X?

![alt text][lda_resample]
[lda_resample]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/lda_resample.png

* Get results

![alt text][lda_results]
[lda_results]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/lda_results.png

* Evaluate model

 __*Hard: Unsupervised learning. No labels*__

 __*Human-in-the-loop*__

    
> Word intrusion

For each trained topic, take first ten words, substitute one of them with another, randomly chosen word (intruder!) and see whether a human can reliably tell which one it was. If so, the trained topic is topically coherent (good); if not, the topic has no discernible theme (bad) [2]

> Topic intrusion: 

Subjects are shown the title and a snippet from a document. Along with the document they are presented with four topics. Three of those topics are the highest probability topics assigned to that document. The remaining intruder topic is chosen randomly from the other low-probability topics in the model

 
![alt text][lda_eva]
[lda_eva]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/lda_eva.png

__*Metrics*__

Cosine similarity: split each document into two parts, and check that topics of the first half are similar to topics of the second halves of different documents are mostly dissimilar

![alt text][lda_metric]
[lda_metric]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/lda_metric.png

More Metrics [reference](http://mimno.infosci.cornell.edu/slides/details.pdf):

1. Size (#	of tokens assigned)
2. Within-doc rank
3. Similarity to corpus-wide distribution
4. Locally-frequent words
5. Co-doc Coherence

## LDA Python Lib

 * Gensim: https://radimrehurek.com/gensim/

 * Graphlab: https://dato.com/products/create/docs/generated/graphlab.topic_model.create.html

 * lda: http://pythonhosted.org//lda/

> Warning: LDA in scikit-learn refers to Linear Discriminant Analysis! scikit-learn implements alternative algorithms, e.g. NMF (Non Negative Matrix Factorization)

* Step 1: [Load Gensim](http://radimrehurek.com/gensim/wiki.html#latent-dirichlet-allocation)

```python
import gensim
# load id->word mapping (the dictionary)
id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
# load corpus iterator
mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
# extract 100 LDA topics, using 20 full passes, (batch mode) no online updates
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=0, passes=20)
```
* Step 2: [Graphlab](https://dato.com/products/create/docs/generated/graphlab.topic_model.create.html)
```python
import graphlab as gl
docs = graphlab.SArray('http://s3.amazonaws.com/dato-datasets/nytimes')
m = gl.topic_model.create(docs,
                          num_topics=20,       # number of topics
                          num_iterations=10,   # algorithm parameters
                          alpha=.01, beta=.1)  # hyperparameters
```

* Step 3: LDA
```python
import lda
X = lda.datasets.load_reuters()
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available
```

## The whole process system

* Pipeline

![alt text][pipeline]
[pipeline]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/pipeline.png

* Pre-Process

![alt text][preprocess]
[preprocess]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/preprocessing.png

* Vector Space

![alt text][vector-space]
[vector-space]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/vector-space.png

* Gensim Model

![alt text][gensim]
[gensim]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/gensim.png

* Evaluation-Virtulization [LDAVis](https://github.com/cpsievert/LDAvis, https://github.com/bmabey/pyLDAvis)

![alt text][ldavis]
[ldavis]: https://github.com/zhangruiskyline/NLP_DeepLearning/blob/master/img/ldavis.png

* Topik

The aim of topik is to provide a full suite and high-level interface for anyone interested in applying topic modeling. For that purpose, topik includes many utilities beyond statistical modeling algorithms and wraps all of its features into an easy callable function and a command line interface.

[https://github.com/ContinuumIO/topik](https://github.com/ContinuumIO/topik)
... automating the pipeline

```python
from topik.run import run_model
run_model('data.json', field='abstract', model='lda_online', r_ldavis=True, output_file=True)
```

## Reference
http://www.cs.princeton.edu/~blei/papers/Blei2012.pdf

http://miriamposner.com/blog/very-basic-strategies-for-interpreting-results-from-the-topic-modeling-tool/

http://journalofdigitalhumanities.org/2-1/topic-modeling-a-basic-introduction-by-megan-r-brett/

https://beta.oreilly.com/ideas/topic-models-past-present-and-future

IPython notebooks explaining Dirichlet Processes, HDPs, and Latent Dirichlet Allocation from Timothy Hopper

https://github.com/tdhopper/notes-on-dirichlet-processes

Slides: 
http://chdoig.github.com/pygotham-topic-modeling

# Section 5: Deep Learning NLP

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

# RNN Intro

## Parse Tree

All words are on leaf nodes.


# Python 3 change

> use dict.items() instead of dict.iteritems()

> use range() instead of xrange()





