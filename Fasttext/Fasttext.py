import os,sys
import pandas as pd
import numpy as np
import fasttext
from urllib.request import urlopen     # For Python 3.0 and later
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print(sys.version)

def plot_words(X, labels, classes=None, xlimits=None, ylimits=None):
    plt.figure(figsize=(6, 6))
    if xlimits is not None:
        plt.xlim(xlimits)
    if ylimits is not None:
        plt.ylim(ylimits)
    plt.scatter(X[:, 0], X[:, 1], c=classes)
    # for i, txt in enumerate(labels):
    #     plt.annotate(txt, (X[i, 0], X[i, 1]))
    plt.show()

def clean_dataset(dataframe, shuffle=False, encode_ascii=False, clean_strings = False, label_prefix='__label__'):
    # Transform train file
    df = dataframe[['name','description']].apply(lambda x: x.str.replace(',',' '))
    df['class'] = label_prefix + dataframe['class'].astype(str) + ' '
    if clean_strings:
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace('"',''))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace('\'',' \' '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace('.',' . '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace('(',' ( '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace(')',' ) '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace('!',' ! '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace('?',' ? '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace(':',' '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.replace(';',' '))
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.lower())
    if shuffle:
        df.sample(frac=1).reset_index(drop=True)
    if encode_ascii :
        df[['name','description']] = df[['name','description']].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii','ignore').str.decode('utf-8'))
    df['name'] = ' ' + df['name'] + ' '
    df['description'] = ' ' + df['description'] + ' '
    return df

def main():
    data_path = '/Users/ruizhang/Documents/NLP_dataset/'


    #############
    #
    ############
    # Load train set
    train_file = data_path +'dbpedia_csv/train.csv'
    df = pd.read_csv(train_file, header=None, names=['class', 'name', 'description'])

    # Load test set
    test_file = data_path + 'dbpedia_csv/test.csv'
    df_test = pd.read_csv(test_file, header=None, names=['class', 'name', 'description'])

    # Mapping from class number to class name
    class_dict = {
        1: 'Company',
        2: 'EducationalInstitution',
        3: 'Artist',
        4: 'Athlete',
        5: 'OfficeHolder',
        6: 'MeanOfTransportation',
        7: 'Building',
        8: 'NaturalPlace',
        9: 'Village',
        10: 'Animal',
        11: 'Plant',
        12: 'Album',
        13: 'Film',
        14: 'WrittenWork'
    }
    df['class_name'] = df['class'].map(class_dict)
    df.head()

    #############
    #
    ############
    desc = df.groupby('class')
    desc.describe().transpose()

    # Transform datasets
    df_train_clean = clean_dataset(df, True, False)
    df_test_clean = clean_dataset(df_test, False, False)

    # Write files to disk
    train_file_clean = data_path + 'dbpedia.train'
    df_train_clean.to_csv(train_file_clean, header=None, index=False, columns=['class', 'name', 'description'])

    test_file_clean = data_path + 'dbpedia.test'
    df_test_clean.to_csv(test_file_clean, header=None, index=False, columns=['class', 'name', 'description'])

    # Train a classifier
    output_file = data_path + 'dp_model'
    classifier = fasttext.supervised(train_file_clean, output_file, label_prefix='__label__')

    result = classifier.test(test_file_clean)
    print('P@1:', result.precision)
    print('R@1:', result.recall)
    print('Number of examples:', result.nexamples)

    sentence1 = ['Picasso was a famous painter born in Malaga, Spain. He revolutionized the art in the 20th century.']
    labels1 = classifier.predict(sentence1)
    class1 = int(labels1[0][0])
    print("Sentence: ", sentence1[0])
    print("Label: %d; label name: %s" % (class1, class_dict[class1]))

    sentence2 = ['One of my favourite tennis players in the world is Rafa Nadal.']
    labels2 = classifier.predict_proba(sentence2)
    class2, prob2 = labels2[0][0]  # it returns class2 as string
    print("Sentence: ", sentence2[0])
    print("Label: %s; label name: %s; certainty: %f" % (class2, class_dict[int(class2)], prob2))

    sentence3 = ['Say what one more time, I dare you, I double-dare you motherfucker!']
    number_responses = 3
    labels3 = classifier.predict_proba(sentence3, k=number_responses)
    print("Sentence: ", sentence3[0])
    for l in range(number_responses):
        class3, prob3 = labels3[0][l]
        print("Label: %s; label name: %s; certainty: %f" % (class3, class_dict[int(class3)], prob3))

    # Load train set
    train_file = data_path + 'amazon_review_polarity_train.csv'
    df_sentiment_train = pd.read_csv(train_file, header=None, names=['class', 'name', 'description'])

    # Load test set
    test_file = data_path + 'amazon_review_polarity_test.csv'
    df_sentiment_test = pd.read_csv(test_file, header=None, names=['class', 'name', 'description'])

    # Transform datasets
    df_train_clean = clean_dataset(df_sentiment_train, True, False)
    df_test_clean = clean_dataset(df_sentiment_test, False, False)

    # Write files to disk
    train_file_clean = data_path + 'amazon.train'
    df_train_clean.to_csv(train_file_clean, header=None, index=False, columns=['class', 'name', 'description'])

    test_file_clean = data_path + 'amazon.test'
    df_test_clean.to_csv(test_file_clean, header=None, index=False, columns=['class', 'name', 'description'])

    dim = 10
    lr = 0.1
    epoch = 5
    min_count = 1
    word_ngrams = 2
    bucket = 10000000
    thread = 12
    label_prefix = '__label__'

    # Train a classifier
    output_file = data_path + 'amazon_model'
    classifier = fasttext.supervised(train_file_clean, output_file, dim=dim, lr=lr, epoch=epoch,
                                     min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,
                                     thread=thread, label_prefix=label_prefix)

    # Evaluate classifier
    result = classifier.test(test_file_clean)
    print('P@1:', result.precision)
    print('R@1:', result.recall)
    print('Number of examples:', result.nexamples)

    class_dict = {
        1: "Negative",
        2: "Positive"
    }

    sentence1 = ["The product design is nice but it's working as expected"]
    labels1 = classifier.predict_proba(sentence1)
    class1, prob1 = labels1[0][0]  # it returns class as string
    print("Sentence: ", sentence1[0])
    # print("Label: %s; label name: %s; certainty: %f" % (class1, class_dict[int(class1)], prob1))

    sentence2 = ["I bought the product a month ago and it was working correctly. But now is not working great"]
    labels2 = classifier.predict_proba(sentence2)
    class2, prob2 = labels2[0][0]  # it returns class as string
    print("Sentence: ", sentence2[0])
    # print("Label: %s; label name: %s; certainty: %f" % (class2, class_dict[int(class2)], prob2))

    url = "https://twitter.com/miguelgfierro/status/805827479139192832"
    response = urlopen(url).read()
    title = str(response).split('<title>')[1].split('</title>')[0]
    print(title)

    # # Format tweet
    # tweet = unescape(title)
    # print(tweet)
    #
    # # Classify tweet
    # label_tweet = classifier.predict_proba([tweet])
    # class_tweet, prob_tweet = label_tweet[0][0]
    # print("Label: %s; label name: %s; certainty: %f" % (class_tweet, class_dict[int(class_tweet)], prob_tweet))


    wiki_dataset_original = data_path + 'enwik9'
    wiki_dataset = data_path + 'text9'
    if not os.path.isfile(wiki_dataset):
        os.system("perl wikifil.pl " + wiki_dataset_original + " > " + wiki_dataset)

    output_skipgram = data_path + 'skipgram'
    if os.path.isfile(output_skipgram + '.bin'):
        skipgram = fasttext.load_model(output_skipgram + '.bin')
    else:
        skipgram = fasttext.skipgram(wiki_dataset, output_skipgram, lr=0.02, dim=50, ws=5,
                                     epoch=1, min_count=5, neg=5, loss='ns', bucket=2000000, minn=3, maxn=6,
                                     thread=4, t=1e-4, lr_update_rate=100)
    print(np.asarray(skipgram['king']))

    print("Number of words in the model: ", len(skipgram.words))

    # Get the vector of some word
    Droyals = np.sqrt(pow(np.asarray(skipgram['king']) - np.asarray(skipgram['queen']), 2)).sum()
    print(Droyals)
    Dpeople = np.sqrt(pow(np.asarray(skipgram['king']) - np.asarray(skipgram['woman']), 2)).sum()
    print(Dpeople)
    Dpeople2 = np.sqrt(pow(np.asarray(skipgram['man']) - np.asarray(skipgram['woman']), 2)).sum()
    print(Dpeople2)

    print(len(skipgram.words))
    targets = ['man', 'woman', 'king', 'queen', 'brother', 'sister', 'father', 'mother', 'grandfather', 'grandmother',
               'cat', 'dog', 'bird', 'squirrel', 'horse', 'pig', 'dove', 'wolf', 'kitten', 'puppy']
    classes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    X_target = []
    for w in targets:
        X_target.append(skipgram[w])
    X_target = np.asarray(X_target)
    word_list = list(skipgram.words)[:10000]
    X_subset = []
    for w in word_list:
        X_subset.append(skipgram[w])
    X_subset = np.asarray(X_subset)
    X_target = np.concatenate((X_subset, X_target))
    print(X_target.shape)
    X_tsne = TSNE(n_components=2, perplexity=40, init='pca', method='exact',
                  random_state=0, n_iter=200, verbose=2).fit_transform(X_target)
    print(X_tsne.shape)
    X_tsne_target = X_tsne[-20:, :]
    print(X_tsne_target.shape)
    plot_words(X_tsne_target, targets, classes=classes)
    plot_words(X_tsne_target, targets, xlimits=[0.5, 0.7], ylimits=[-3.7, -3.6])

if __name__ == '__main__':
    main()