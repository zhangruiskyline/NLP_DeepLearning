import math
from textblob import TextBlob as tb



def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

document1 = tb("""Python is a 2000 made-for-TV horror movie directed by Richard
Clabaugh. The film features several cult favorite actors, including William
Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy,
Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the
A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean
Whalen. The film concerns a genetically engineered snake, a python, that
escapes and unleashes itself on a small town. It includes the classic final
girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles,
 California and Malibu, California. Python was followed by two sequels: Python
 II (2002) and Boa vs. Python (2004), both also made-for-TV films.""")

document2 = tb("""Python, from the Greek word (πύθων/πύθωνας), is a genus of
nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are
recognised.[2] A member of this genus, P. reticulatus, is among the longest
snakes known.""")

document3 = tb("""Thank you, Ashish. Good afternoon everyone. Well, we delivered strong financial results for the first quarter with revenue of $41.5 billion and gross margin at 62.4%, both at a very top end of our guidance.
Earnings per share of $3.63 grew by 5% sequentially, while net revenue was essentially flat. Revenue was better than expected in all four segments. The benefits we achieved through business diversification clearly came through this quarter with growth in Wired, Enterprise Storage, and Industrial completely offsetting the typical negative seasonality from Wireless.

The integration of classic Broadcom has gone very well and is now mostly complete. We remain focused on driving financial performance towards our long-term operating margin and free cash flow targets.

Let me now turn to a discussion of our segment results, starting with Wired, our largest segment. In the first quarter, Wired revenue came in at $2.09 billion better than expected and represented 50% of our total revenue. Revenue for this segment was up slightly on a sequential basis, benefited from strong demand for Ethernet Switching and Routing products from cloud data center operators.

This growth was partially offset by the continuing seasonal decline in demand for our broadband carrier access and set-top box products, which we expect to bottom in this first quarter.

Turning to the second fiscal quarter, we forecast Wired revenue experience sequential growth a little bit stronger than what we saw in the prior quarter. We expect the momentum from cloud data center demand to sustain and expect a seasonal increase in demand for our broadband access and set-top box products.""")

bloblist = [document1, document2, document3]
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

my_additional_stop_words = "good afternoon, quater, well"

stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)
corpus = ["""Thank you, Ashish. Good afternoon everyone. Well, we delivered strong financial results for the first quarter with revenue of $41.5 billion and gross margin at 62.4%, both at a very top end of our guidance.
Earnings per share of $3.63 grew by 5% sequentially, while net revenue was essentially flat. Revenue was better than expected in all four segments. The benefits we achieved through business diversification clearly came through this quarter with growth in Wired, Enterprise Storage, and Industrial completely offsetting the typical negative seasonality from Wireless.

The integration of classic Broadcom has gone very well and is now mostly complete. We remain focused on driving financial performance towards our long-term operating margin and free cash flow targets.

Let me now turn to a discussion of our segment results, starting with Wired, our largest segment. In the first quarter, Wired revenue came in at $2.09 billion better than expected and represented 50% of our total revenue. Revenue for this segment was up slightly on a sequential basis, benefited from strong demand for Ethernet Switching and Routing products from cloud data center operators.

This growth was partially offset by the continuing seasonal decline in demand for our broadband carrier access and set-top box products, which we expect to bottom in this first quarter.

Turning to the second fiscal quarter, we forecast Wired revenue experience sequential growth a little bit stronger than what we saw in the prior quarter. We expect the momentum from cloud data center demand to sustain and expect a seasonal increase in demand for our broadband access and set-top box products."""]
vectorizer = TfidfVectorizer(analyzer=u'word',max_df=1,lowercase=True,stop_words=set(stop_words),max_features=15)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
print (dict(zip(vectorizer.get_feature_names(), idf)))



from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


train_set = ["The sky is blue.", "The sun is bright.", "The sun in the sky is bright."]
stop_words = stopwords.words('english')

transformer = TfidfVectorizer(stop_words=stop_words)
transformer.fit_transform(train_set).todense()

test_set = ["sky","land","sea","water","sun","moon"] #Query
transformer.transform(test_set).todense()
transformer = TfidfVectorizer(stop_words=stop_words, vocabulary=test_set)
transformer.fit_transform(train_set).todense().T

print (tfidf.todense())

