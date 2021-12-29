import nltk
import re
import pandas as pd
import numpy as np

from nltk import word_tokenize
from collections import defaultdict
from collections import Counter

from nltk.corpus import inaugural
from nltk.corpus import gutenberg
nltk.download('inaugural') 	# Note: on 12/4 nltk.download('inaugural') stopped working. On 12/7 it worked again.
				# Code to work around this is commented out
nltk.download('stopwords')

# To show list of inaugurals
# inaugural.fileids()

# Help from https://www.nltk.org/book/ch02.html

text1 = inaugural.raw('1861-Lincoln.txt')
text2 = inaugural.raw('1905-Roosevelt.txt')
text3 = inaugural.raw('1933-Roosevelt.txt')
text4 = inaugural.raw('1961-Kennedy.txt')
text5 = inaugural.raw('1981-Reagan.txt')
text6 = inaugural.raw('2009-Obama.txt')
text7 = inaugural.raw('2017-Trump.txt')

# If files are manually uploaded to google colab:
# from google.colab import files
# uploaded = files.upload()

# file_name1 = "1861-Lincoln.txt"
# file_name2 = "1905-Roosevelt.txt"
# file_name3 = "1933-Roosevelt.txt"
# file_name4 = "1961-Kennedy.txt"
# file_name5 = "1981-Reagan.txt"
# file_name6 = "2009-Obama.txt"
# file_name7 = "2017-Trump.txt"
# file_name8 = "2021-Biden.txt"
# text1 = uploaded[file_name1].decode("utf-8")
# text2 = uploaded[file_name2].decode("utf-8")
# text3 = uploaded[file_name3].decode("utf-8")
# text4 = uploaded[file_name4].decode("utf-8")
# text5 = uploaded[file_name5].decode("utf-8")
# text6 = uploaded[file_name6].decode("utf-8")
# text7 = uploaded[file_name7].decode("utf-8")
# text8 = uploaded[file_name8].decode("utf-8")

text8_raw = open("2021-Biden.txt")
text8 = text8_raw.read() # Lab: '.read()', also works best for normalize_document
text8_raw.close()

corpus = [text1, text2, text3, text4, text5, text6, text7, text8]

# Function that will tokenize documents
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
def normalize_document(doc):
    # lowercase and remove special characters\whitespace
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc
# Code to go through and tokenize documents in a corpus
normalize_corpus = np.vectorize(normalize_document)

norm_corpus = normalize_corpus(corpus)

# Tfidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(min_df=0., max_df=1., norm="l2",
                     use_idf=True, smooth_idf=True)
tv_matrix = tv.fit_transform(norm_corpus)
tv_matrix = tv_matrix.toarray()
# vocab = tv.get_feature_names()
# pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)

# For Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
# get bag of words features in sparse format
cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)

# get all unique words in the corpus
vocab = cv.get_feature_names_out()
# show document feature vectors
# pd.DataFrame(cv_matrix, columns=vocab)

from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)
similarity_df = similarity_df.set_axis(['Lincoln', 'T. Roosevelt', 'F. Roosevelt', 
                        'Kennedy', 'Reagan', 'Obama', 'Trump', 'Biden'], axis = 1, inplace = False)
similarity_df = similarity_df.set_axis(['Lincoln', 'T. Roosevelt', 'F. Roosevelt', 
                        'Kennedy', 'Reagan', 'Obama', 'Trump', 'Biden'], axis = 0, inplace = False)
print(similarity_df)
print()
print("Similarity appears to go by date. Obama's speech appears to be more simlar to the all the others. Biden's to Obama's is the most similar of two speeches.")
print()
print("Performing LDA analysis to form topics based on tokens in the inaugural addresses:")
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=3, max_iter=10000, random_state=0)
dt_matrix = lda.fit_transform(cv_matrix)
features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3'])
features = features.set_axis(['Lincoln', 'T. Roosevelt', 'F. Roosevelt', 
                        'Kennedy', 'Reagan', 'Obama', 'Trump', 'Biden'], axis = 0, inplace = False)

print()
print(features)
print()
# note: had a 38s run time

tt_matrix = lda.components_
for topic_weights in tt_matrix:
    topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]
    topic = sorted(topic, key=lambda x: -x[1])
    topic = [item for item in topic if item[1] > 15.0]
    print(topic)
    print()
print("Though many similar words are used, to pick up on the importance of certain words and my understanding of history, topic 1 could be about America and who America is. Topic 2 would be about the purpose government and America's place in the world. Topic 3 fits best with Lincoln's address, which was a special time of looking at the constitution and what that said about a union that was threatening to divide.")
print()
print("LDA USING BIGRAMS")
print("Looking to see how topics change if bigrams are used:")
print()

# Creating a matrix for bigrams from the addresses
bv = CountVectorizer(ngram_range=(2,2))
bv_matrix = bv.fit_transform(norm_corpus)
bv_matrix = bv_matrix.toarray()
vocab = bv.get_feature_names_out()
pd.DataFrame(bv_matrix, columns=vocab)

from sklearn.decomposition import LatentDirichletAllocation
blda = LatentDirichletAllocation(n_components=4, max_iter=10000, random_state=0)
bt_matrix = blda.fit_transform(bv_matrix)
features = pd.DataFrame(bt_matrix, columns=['T1', 'T2', 'T3', 'T4'])
features = features.set_axis(['Lincoln', 'T. Roosevelt', 'F. Roosevelt', 
                        'Kennedy', 'Reagan', 'Obama', 'Trump', 'Biden'], axis = 0, inplace = False)
print(features)
print()
# note: had a 41s run time

btt_matrix = blda.components_
for topic_weights in btt_matrix:
    topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]
    topic = sorted(topic, key=lambda x: -x[1])
    topic = [item for item in topic if item[1] > 4.0]
    print(topic)
    print()
print("Broadly speaking here, topic 1 has to do with contry unity and future. Topic 2 appears to be a call to work together. Topic 3 also appears to be looking toward the future with a request for blessing from God. Topic 4, again most typified by the Lincoln speech, is about the united states and the importance of its constitution.")
print()
print("Given more time, I would like to do this similar type of analysis only with all the addresses to see,  generally, what presidents talk about in their inaugurals. This was more of a look at the first inaugural addresses of distinct and recent presidents.")
print()

# Major reference for help with this code comes from Text Analytics with Python by Dipanjan Sarkar
# Other help came from https://www.nltk.org/book/ch02.html for using the nltk library
