from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
#import pickle

def model_fit(number_topics, number_words, subject_data):
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(subject_data.dropna())
    dic=count_vectorizer.get_feature_names()
    lda = LDA(n_components=number_topics)
    model = (dic,lda.components_,lda.exp_dirichlet_component_,lda.doc_topic_prior_)
    return model
    #with open('/content/outfile', 'wb') as fp:
    #    pickle.dump(model, fp)

def model_predict(model,prints=True,data_samples):
    lda = LDA()
    (features,lda.components_,lda.exp_dirichlet_component_,lda.doc_topic_prior_)=model
    tf_vectorizer = CountVectorizer(vocabulary=features)
    tf = tf_vectorizer.fit_transform(data_samples)
    predict = lda.transform(tf)
    #print(predict)
    if prints==True:
        print("Topics found via LDA:")
        print_topics(lda, tf_vectorizer, number_words)
    return predict

def print_topics(model, tf_vectorizer, n_top_words):
    words = tf_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))