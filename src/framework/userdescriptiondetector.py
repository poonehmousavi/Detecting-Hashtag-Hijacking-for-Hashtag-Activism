import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


class UserDescriptionDetector:

    def __init__(self, model_filePath):
        self.model_filePath= model_filePath
        self.model=self.loadModel(model_filePath)

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
        loaded_model=None
        try:
            loaded_model = pickle.load(open(filePath, 'rb'))
        except FileNotFoundError:
            pass
            # print('There is no exixiting model')
        return  loaded_model

    def train_model(self,X,Y):
        vect, clf= self.build_model('tfidf')
        clf.fit(X, Y)
        self.model=clf
        self.save_model(clf,self.model_filePath)
        return  clf



    def build_model(self,mode):
        vect = None
        if mode == 'count':
            vect = CountVectorizer()
        elif mode == 'tf':
            vect = TfidfVectorizer(use_idf=False, norm='l2')
        elif mode == 'tfidf':
            vect = TfidfVectorizer()
        else:
            raise ValueError('Mode should be either count or tfidf')

        return Pipeline([
            ('vect', vect),
            ('clf', LogisticRegression(solver='newton-cg', n_jobs=-1))
        ])


    def pipeline(self,x, y, mode):
        processed_x =x['user.description']

        model_pipeline = self.build_model(mode)
        cv = KFold(n_splits=5, shuffle=True)

        scores = cross_val_score(model_pipeline, processed_x, y, cv=cv, scoring='accuracy')
        print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

        return model_pipeline



    def predict(self,tweets,prob=True):
        if prob== False:
            no_desc = tweets[pd.isna(tweets['user.description'])]
            no_desc['ud_label'] = 1
            with_dec = tweets[pd.notna(tweets['user.description'])]
            X = with_dec['user.description']
            with_dec['ud_label'] = self.model.predict(X)
            result = no_desc.append(with_dec)
            return  result
        else:
            no_desc = tweets[pd.isna(tweets['user.description'])]
            no_desc['ud_label'] = 1
            with_dec = tweets[pd.notna(tweets['user.description'])]
            X = with_dec['user.description']
            if X.shape[0] !=0:
                with_dec['ud_label'] = self.model.predict_proba(X)[:, 1]
                result = no_desc.append(with_dec)
            else:
                result=no_desc
            return result



    def fit(self,training,save=False):
        training['ud_label'] = 0
        no_desc = training[pd.isna(training['user.description'])]
        # print('number of tweets without user description is %d ' % no_desc.shape[0])
        with_desc = training[pd.notna(training['user.description'])]
        # print('number of tweets with user description is %d ' % with_desc.shape[0])
        x = with_desc['user.description']
        y = with_desc['label']

        model_pipeline = self.build_model('tfidf')
        model_pipeline.fit(x, y)
        self.model = model_pipeline
        if (save==True):
            self.save_model(self.model,self.model_filePath)

    def fit_sub(self, x, y,save=False):
        model_pipeline = self.build_model('tfidf')
        model_pipeline.fit(x, y)
        self.model = model_pipeline
        if (save == True):
            self.save_model(self.model, self.model_filePath)
        return self.model


    def save_model(self,model,file_path):
        pickle.dump(model, open(file_path, 'wb'))

    def evaluate_predictions(self, Y_pred, Y_true, pos_label=1):
        pred = [1 if n > 0.5 else 0 for n in Y_pred]
        precision = precision_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        recall = recall_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        fmeasure = f1_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        # auc_score = auc(recall, precision)
        roc_score = roc_auc_score(Y_true, Y_pred)

        return (roc_score,precision, recall, fmeasure)

