import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
#import pyLDAvis
#import pyLDAvis.gensim
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def preprocessing(data_words):
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) 
        if word not in stop_words] for doc in texts]
    
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    
    def make_trigrams(texts):
        [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = make_bigrams(data_words_nostops)
    nlp = spacy.load('en', disable=['parser', 'ner'])
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    id2word = corpora.Dictionary(data_lemmatized)
    texts = data_lemmatized
    corpus = [id2word.doc2bow(text) for text in texts]
    return corpus,id2word,texts

def model_fit(data_words,num_topics=3):
    corpus,id2word,texts = preprocessing(data_words)
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, 
        update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    lda_model.save('./lda_train.model')
    return lda_model

def model_predict(lda_model,prints=True,data):
    corpus,id2word,texts = preprocessing(data)
    train_vecs = []
    for i in range(len(texts)):
        top_topics = (lda_model.get_document_topics(corpus[i],minimum_probability=0.0))
        topic_vec = [top_topics[i][1] for i in range(3)]
        train_vecs.append(topic_vec)
    return train_vecs