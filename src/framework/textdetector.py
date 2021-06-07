import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score, precision_score, recall_score, \
    precision_recall_curve

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import SGDClassifier

class TextDetector():
    """"
    Clustering step of semi-supervised module
    """

    def __init__(self, model_filePath):
        self.model_filepath = model_filePath
        self.model = self.loadModel(model_filePath)


    def loadModel(self, filePath):
        try:
            loaded_model = pickle.load(open(filePath, 'rb'))
            return loaded_model
        except FileNotFoundError:
            pass
            # print('No Model exists')

    def make_pipeline(self,loss='hing'):
        pipeline_sgd = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('nb', SGDClassifier(loss=loss)),
        ])
        return  pipeline_sgd

    def fit(self,training,save=False):
        X_train = training['processed_fulltext']
        y_train = training['label']

        model_pipeline = self.make_pipeline(loss='log')
        model_pipeline.fit(X_train, y_train)
        self.model = model_pipeline
        if (save==True):
            self.save_model(self.model,self.model_filepath)
        return self.model


    def fit_sub(self,X_train,y_train,save=False):
        model_pipeline = self.make_pipeline(loss='log')
        model_pipeline.fit(X_train, y_train)
        self.model = model_pipeline
        if (save==True):
            self.save_model(self.model,self.model_filepath)
        return self.model

    def predict(self,test,prob=True):
        X_test = test['processed_fulltext']
        if prob== False:
            y_predict = self.model.predict(X_test)

        else:
            y_predict = self.model.predict_proba(X_test)[:, 1]

        return y_predict

    def save_model(self,model,file_path):
        pickle.dump(model, open(file_path, 'wb'))

