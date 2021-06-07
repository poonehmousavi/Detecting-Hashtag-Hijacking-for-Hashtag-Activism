from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, auc
# from wordcloud import WordCloud
import pandas as pd
from boto import sns
from gensim.models import TfidfModel
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import gensim.models as models
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sns.set_style('whitegrid')

import numpy as np  # linear algebra
import pandas as pd  # report processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.manifold import TSNE

stop_words = nltk.corpus.stopwords.words('english')
newStopWords = ["https", "metoo", "com", "amp"]
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "%"]
stop_words.extend(newStopWords)
stop_words.extend(punc)


class ClusterDetector():
    """"
    Clustering step of semi-supervised module
    """

    def __init__(self, km_model_filepath, vector_filepath, cluster_label_filepath, threshold):
        self.threshold = threshold
        self.km_model_filepath = km_model_filepath
        self.vector_filepath = vector_filepath
        self.cluster_label_filepath = cluster_label_filepath
        self.km_model = self.loadModel(km_model_filepath)
        self.vector = self.loadModel(vector_filepath)
        self.cluster_labels = self.load_data(cluster_label_filepath + 'label.json')
        self.cluster_prob = self.load_data(cluster_label_filepath + 'prob.json')

    def loadModel(self, filePath):
        try:
            loaded_model = pickle.load(open(filePath, 'rb'))
            return loaded_model
        except FileNotFoundError:
            pass
            # print('No Model exists')

    def load_vector(self, filePath):
        try:
            loaded_vector_vocab = pickle.load(open(filePath, 'rb'))
            vector = TfidfVectorizer(lowercase=True, stop_words=stop_words, tokenizer=self.tokenize)
            vector.vocabulary_ = loaded_vector_vocab
            return vector
        except FileNotFoundError:
            pass
            # print('No Model exists')

    def load_data(self, filepath):
        try:
            return pd.read_json(filepath, orient='records', lines=True)
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            pass
            # print('Decoding JSON has failed')
        except FileNotFoundError:
            pass
            # print('No File')

    def tokenize(self, sentence):
        return simple_preprocess(str(sentence), deacc=True, min_len=1, max_len=20)

    def get_tfid(self, data):
        vector = TfidfVectorizer(lowercase=True, stop_words=stop_words, tokenizer=self.tokenize)
        tf_idf = vector.fit_transform(data['processed_fulltext'])
        return vector, tf_idf

    def fit(self, training, save=False):
        vector, tf_idf = self.get_tfid(training)
        result = self.create_model(13, vector, tf_idf, training, save=save)
        self.label_clusters(result, save=save)

    def create_model(self, cluster_number, vector, tf_idf, data, save=False):

        km = KMeans(n_clusters=cluster_number)
        clusters = km.fit(tf_idf)

        word_features = vector.get_feature_names()
        if (save == True):
            # print("Top terms per cluster %s:", str(cluster_number))
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]
            terms = vector.get_feature_names()
            common_words = km.cluster_centers_.argsort()[:, -1:-26:-1]
            # for num, centroid in enumerate(common_words):
            #     print(str(num) + ' : ' + ', '.join(word_features[word] for word in centroid))

        data['cluster'] = km.labels_
        # pickle.dump(vector.vocabulary_, open("feature.pkl", "wb"))
        #
        self.vector = vector
        self.km_model = km
        if (save == True):
            self.save_model(vector, self.vector_filepath)
            self.save_model(km, self.km_model_filepath)

        return data

    def cluster_tweets(self, data):
        tf_idf = self.vector.transform(data["processed_fulltext"])
        data["cluster"] = self.km_model.predict(tf_idf).reshape(-1, 1)
        return data

    def label_clusters(self, data, save=False):

        clusters = data.groupby('cluster').label.apply(lambda x: self.majority_vote(x)).reset_index(0)
        clusters.rename(columns={'label': 'cl_label'}, inplace=True)
        self.cluster_labels = clusters
        if (save == True):
            clusters.to_json(self.cluster_label_filepath + 'label.json', orient='records', lines=True)

        clusters = data.groupby('cluster').label.apply(lambda x: self.weighted_vote(x)).reset_index(0)
        clusters.rename(columns={'label': 'cl_label'}, inplace=True)
        self.cluster_prob = clusters
        if (save == True):
            clusters.to_json(self.cluster_label_filepath + 'prob.json', orient='records', lines=True)

    def majority_vote(self, grp):
        counts = grp.value_counts(normalize=True).to_dict()

        valid_count = counts.get(0)
        hijacked_count = counts.get(1)

        if valid_count != None and valid_count >= self.threshold:
            return 0
        elif hijacked_count != None and hijacked_count >= self.threshold:
            return 1
        else:
            return -1

    def weighted_vote(self, grp):
        counts = grp.value_counts(normalize=True).to_dict()

        valid_count = counts.get(0)
        hijacked_count = counts.get(1)

        if valid_count != None and valid_count >= self.threshold:
            return 1 - valid_count
        elif hijacked_count != None and hijacked_count >= self.threshold:
            return hijacked_count
        else:
            return 0.5

    def predict(self, data, prob=True):
        if prob == False:
            data = self.cluster_tweets(data)
            result = pd.merge(data,
                              self.cluster_labels,
                              on='cluster')
            return result
        else:
            data = self.cluster_tweets(data)
            result = pd.merge(data,
                              self.cluster_prob,
                              on='cluster')
            return result

    def save_model(self, model, file_path):
        pickle.dump(model, open(file_path, 'wb'))

    def evaluate_predictions(self, Y_pred, Y_true, pos_label=1):
        pred = [1 if n > 0.5 else 0 for n in Y_pred]
        precision = precision_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        recall = recall_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        fmeasure = f1_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        # auc_score = auc(recall, precision)
        roc_score = roc_auc_score(Y_true, Y_pred)

        return (roc_score,precision, recall, fmeasure)


