
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np
import pickle
from collections import Counter
import pandas as pd

feature_dic={}

def loadData(file_path):
    data =  pd.read_json(file_path,orient='records', lines=True)
    return  data

class SupervisedDetector():
    """
    Supevised step of semi- supervised module
    """

    def __init__(self,rf_model_filepath,lr_model_filepath,nb_model_filepath):
        self.rf_model_filepath=rf_model_filepath
        self.rf_model=self.loadModel(rf_model_filepath)
        self.lr_model_filepath=lr_model_filepath
        self.lr_model=self.loadModel(lr_model_filepath)
        self.nb_model_filepath=nb_model_filepath
        self.nb_model=self.loadModel(nb_model_filepath)



    def load_data(self, filepath):
        try:
            return pd.read_json(filepath, orient='records', lines=True)
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            pass
            # print('Decoding JSON has failed')
        except FileNotFoundError:
            pass
            # print('No File')


    def loadModel(self,filePath):
        try:
            loaded_model = pickle.load(open(filePath, 'rb'))
            return loaded_model
        except FileNotFoundError:
            pass
            # print('No Model exists')


    def extract_features(self,data):
        """
        Create features and normalize them.
        """
        feature_cols = ['user.friends_count', 'user.followers_count','retweet_count','favorite_count','user.verified','number_of_hashtags']
        # Normalize total_bedrooms column

        data= data.loc[:, feature_cols]
        normalized_df=data;
        normalized_df['user.friends_count'] = self.normalize(normalized_df,'user.friends_count')
        normalized_df['user.followers_count'] = (normalized_df['user.followers_count'] - normalized_df['user.followers_count'].min()) / (normalized_df['user.followers_count'].max() - normalized_df['user.followers_count'].min())
        normalized_df['retweet_count'] = (normalized_df['retweet_count'] - normalized_df['retweet_count'].min()) / (normalized_df['retweet_count'].max() - normalized_df['retweet_count'].min())
        normalized_df['favorite_count'] = (normalized_df['favorite_count'] - normalized_df['favorite_count'].min()) / (normalized_df['favorite_count'].max() - normalized_df['favorite_count'].min())
        # normalized_df=normalized_df.dropna()
        # Y=self.extract_labels(normalized_df)
        # normalized_df= normalized_df.drop(columns=['id','category'])

        return normalized_df


    def normalize(self,data,feature):
        min= data[feature].min()
        max= data[feature].max()
        if min==max:
            return data[feature]
        else:
            return (data[feature] - min) / (
                    max - min)
    def extract_labels(self,data):
        Y =data['label']
        return  Y


    def train_LogisticRegression(self,X,Y):
        logreg = LogisticRegression()
        logreg.fit(X, Y)
        self.lr_model=logreg
        return  logreg


    def train_NaiveBayes(self,X,Y):
        gnb = GaussianNB()
        gnb.fit(X, Y)
        self.nb_model=gnb
        return  gnb

    def train_randomForest(self,X,Y):
         rf= RandomForestClassifier(n_estimators=100, max_depth=2, random_state = 0)
         rf.fit(X, Y)
         self.rf_model=rf
         return  rf

    def predict(self,model,X_Test,prob=True):
        if prob==False:
            Y_test = model.predict(X_Test)
            return  Y_test
        else:
            Y_test = model.predict_proba(X_Test)
            return Y_test






    def save_model(self,model,file_path):
        pickle.dump(model, open(file_path, 'wb'))

    def label_confident_tweets(self,data):
        for index, tweet in data.iterrows():
            if tweet['y_lr']==tweet['y_nb'] and tweet['y_rf']== tweet['y_lr']:
                data.loc[index, "label"] = tweet['y_lr']


    def top_features(self,logreg_model, k):
        W = logreg_model.coef_[0]
        l = []
        for i in range(len(W)):
            l.append((i, W[i]))
        sorted_by_weight = sorted(l, key=lambda tup: abs(float(tup[1])), reverse=True)
        top = []
        for i in range(k):
            temp = [word for word, index in feature_dic.items() if index == sorted_by_weight[i][0]]
            top.append((temp[0], sorted_by_weight[i][1]))

        return top

    def evaluate_predictions(self, Y_pred, Y_true, pos_label=1):
        pred = [1 if n > 0.5 else 0 for n in Y_pred]
        precision = precision_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        recall = recall_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        fmeasure = f1_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        # auc_score = auc(recall, precision)
        roc_score = roc_auc_score(Y_true, Y_pred)

        return (roc_score,precision, recall, fmeasure)

    def fit(self,training,save=False):
        X_train = self.extract_features(training)
        Y_train = self.extract_labels(training)

        logreg = self.train_LogisticRegression(X_train, Y_train)
        if (save == True):
            self.save_model(logreg, self.lr_model_filepath)

        ####################     traing Naive BAyes     #########################
        gnb = self.train_NaiveBayes(X_train, Y_train)
        if (save == True):
            self.save_model(gnb, self.nb_model_filepath)

        ###################     traing Random Forest    #########################
        rf = self.train_randomForest(X_train, Y_train)
        if (save == True):
            self.save_model(rf, self.rf_model_filepath)
        return  logreg,rf,gnb

    def fit_sub(self,X_train,Y_train,save=False):

        logreg = self.train_LogisticRegression(X_train, Y_train)
        if (save == True):
            self.save_model(logreg, self.lr_model_filepath)

        ####################     traing Naive BAyes     #########################
        gnb = self.train_NaiveBayes(X_train, Y_train)
        if (save == True):
            self.save_model(gnb, self.nb_model_filepath)

        ###################     traing Random Forest    #########################
        rf = self.train_randomForest(X_train, Y_train)
        if (save == True):
            self.save_model(rf, self.rf_model_filepath)
        return  logreg,rf,gnb





def check_confident_tweets(tweets):
    return tweets


