import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pyLDAvis.gensim


def main():
    ##dats set from https://www.kaggle.com/aaron7sun/stocknews
    data_path = '/Users/ruizhang/Documents/NLP_dataset/'
    data = pd.read_csv(data_path + "Combined_News_DJIA.csv")

    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']
    print(len(test))
    print(len(train))
    trainheadlines = []
    for row in range(0, len(train.index)):
        trainheadlines.append(' '.join(str(x) for x in train.iloc[row, 2:27]))
    ##Convert a collection of text documents to a matrix of token counts
    trainvect = CountVectorizer()
    ## Transform into a document-term matrix
    Trainfeature = trainvect.fit_transform(trainheadlines)
    ####Detailed view of Document Count Matrix
    DTM_With_Colm = pd.DataFrame(Trainfeature.toarray(), columns=trainvect.get_feature_names())
    Trainfeature.shape

    ## Use regression

    Logis = LogisticRegression()
    Model1 = Logis.fit(Trainfeature, train['Label'])
    testheadlines = []
    for row in range(0, len(test.index)):
        testheadlines.append(' '.join(str(x) for x in test.iloc[row, 2:27]))

    Testfeature = trainvect.transform(testheadlines)
    Testfeature.shape
    Predicted = Model1.predict(Testfeature)
    pd.crosstab(test["Label"], Predicted, rownames=["Actual"], colnames=["Predict"])


    ##Naive bayes
    Nb = MultinomialNB()
    Model2 = Nb.fit(Trainfeature, train['Label'])
    Nbpredicted = Model2.predict(Testfeature)
    pd.crosstab(test["Label"], Nbpredicted, rownames=["Acutal"], colnames=["Predict"])

    y_NaviBayes = Nbpredicted
    y_true = test["Label"]
    accuracy_score(y_NaviBayes, y_true)
    x_Logist = Predicted
    x_true = test["Label"]
    accuracy_score(x_Logist, x_true)

    ##NB-Ngram model
    advvect = CountVectorizer(ngram_range=(1, 2))
    advancedtrain = advvect.fit_transform(trainheadlines)
    advancedtrain.shape
    advmodel = MultinomialNB()
    advancemodel = advmodel.fit(advancedtrain, train["Label"])
    advancetest = advvect.transform(testheadlines)
    advNBprediction = advmodel.predict(advancetest)
    advNBprediction.shape
    pd.crosstab(test["Label"], advNBprediction, rownames=["Acutal"], colnames=["Predicted"])
    x_adNB = advNBprediction
    x_test = test["Label"]
    accuracy_score(x_test, x_adNB)

    ##LDA model
    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list
    en_stop = get_stop_words('en')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # Our Document
    trainheadlines

    # list for tokenized documents in loop
    texts = []

    # loop through document list
    for i in trainheadlines:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # add tokens to list
        texts.append(stemmed_tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=1, chunksize=10000,
                                               update_every=1)
    print(ldamodel.print_topics(num_topics=10, num_words=3))
    ldamodel.print_topics(5)
    ## used in ipython
    ##pyLDAvis.enable_notebook()
    news = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    pyLDAvis.save_html(news, 'lda_stock.html')

if __name__ == '__main__':
    main()