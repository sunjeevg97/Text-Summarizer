from nltk import *
from goose import Goose
from requests import get
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sys import version_info
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import networkx as nx
python3 = version_info[0] > 2

if python3:
    response = input("Please enter a url to scrape")
else:
    response = raw_input("Please enter a url to scrape")

#extract content from inputted url
def extract_content(response):
    url = get(response)
    extractor = Goose()
    article = extractor.extract(raw_html = url.content)
    text = article.cleaned_text
    return text

def num_words(sentences):
    vectorizer = CountVectorizer(min_df = 1)
    matrix = vectorizer.fit_transform(sentences)
    return matrix

#implementation using pagerank -> unsupervised summarization
def text_rank(response):
    text = extract_content(response)
    text = ' '.join(text.strip().split('\n'))
    sent_tokenize = PunktSentenceTokenizer()
    sent_list = sent_tokenize.tokenize(text)
    bag_of_words = num_words(sent_list)

    #normalize using tf-idf
    norm = TfidfTransformer().fit_transform(bag_of_words)
    graph = norm * norm.T
    nx_graph = nx.from_scipy_sparse_matrix(graph)
    rank = nx.pagerank(nx_graph)
    text = sorted(((rank[i],s) for i,s in enumerate(sent_list)),
                  reverse=True)

print text_rank(response)
