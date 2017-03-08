from os import path
from wordcloud import WordCloud
import pandas as pd

d = path.dirname(__file__)

# Read the whole text.

text = "Just two weeks ago, I wrote an article on Broadcom (NASDAQ:AVGO) in which I explained why I had bought shares in a stock at its all-time high. In short the article can be summarized by its title: 3 reasons why I have bought Broadcom: growth, value and prospects"
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud)
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


