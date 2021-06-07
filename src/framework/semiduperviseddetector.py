import pickle
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


from src.framework.preprocessor import Preprocessor
from src.framework.blacklistdetector import BlackListUserDetector
from src.framework.textdetector import TextDetector
from src.framework.whitelistdetector import WhiteListDetector
from src.framework.clusterdetector import ClusterDetector
from src.framework.superviseddetector import SupervisedDetector
from src.framework.userdescriptiondetector import UserDescriptionDetector
import numpy as np


class SemiSupervisedDetector():
    def __init__(self, file_path):
        self.preprocessor= Preprocessor()
        self.blacklist_detector = BlackListUserDetector(blackListUser_filePath=file_path + "user_blackList.json",
                                                   userDictionary_filepath=file_path + "user_hijacked_tweets_counts.json",
                                                   threshhold=5)
        self.whitelist_detector = WhiteListDetector(whiteListUser_filePath=file_path + "user_whiteList.json",
                                               userDictionary_filepath=file_path + "user_vaid_tweets_counts.json",
                                               hijackedDictionary_filepath=file_path + "hijacked_dict.json",
                                               validDictionary_filepath=file_path + "valid_dict.json",
                                               user_threshhold=8, word_threshhold=0.6)
        # self.cluster_detector = ClusterDetector(file_path + "km_Model.sav", file_path + "vector.pkl",
        #                                    file_path + "cluster_", threshold=0.6)
        self.supervised_detector = SupervisedDetector(rf_model_filepath=file_path + "rf_Model.sav",
                                                 lr_model_filepath=file_path + "lr_Model.sav",
                                                 nb_model_filepath=file_path + "nb_Model.sav")
        self.user_desc_detector = UserDescriptionDetector(model_filePath=file_path + "userDesc_count_Model.sav")
        self.text_detector = TextDetector(model_filePath=file_path + "text_Model.sav")
        self.meta_learner=self.loadModel(file_path+"meta_learner.sav")
        self.file_path=file_path


    def loadModel(self,filePath):
        try:
            loaded_model = pickle.load(open(filePath, 'rb'))
            return loaded_model
        except FileNotFoundError:
            print('No Model exists')

    def reload(self):
        file_path=self.file_path
        self.blacklist_detector = BlackListUserDetector(blackListUser_filePath=file_path + "user_blackList.json",
                                                   userDictionary_filepath=file_path + "user_hijacked_tweets_counts.json",
                                                   threshhold=5)
        self.whitelist_detector = WhiteListDetector(whiteListUser_filePath=file_path + "user_whiteList.json",
                                               userDictionary_filepath=file_path + "user_vaid_tweets_counts.json",
                                               hijackedDictionary_filepath=file_path + "hijacked_dict.json",
                                               validDictionary_filepath=file_path + "valid_dict.json",
                                               user_threshhold=8, word_threshhold=0.6)
        # self.cluster_detector = ClusterDetector(file_path + "km_Model.sav", file_path + "vector.pkl",
        #                                    file_path + "cluster_", threshold=0.6)
        self.supervised_detector = SupervisedDetector(rf_model_filepath=file_path + "rf_Model.sav",
                                                 lr_model_filepath=file_path + "lr_Model.sav",
                                                 nb_model_filepath=file_path + "nb_Model.sav")
        self.user_desc_detector = UserDescriptionDetector(model_filePath=file_path + "userDesc_count_Model.sav")
        self.text_detector = TextDetector(model_filePath=file_path + "text_Model.sav")
        self.meta_learner=self.loadModel(file_path+"meta_learner.sav")


    def preprocess_data(self,tweets):
        tweets=self.preprocessor.remove_urls(tweets)
        tweets=self.preprocessor.remove_stop_words(tweets)
        tweets=self.preprocessor.remove_emoji(tweets)
        tweets=self.preprocessor.remove_sign(tweets)
        tweets=self.preprocessor.replace_special_char_abb(tweets)
        tweets=self.preprocessor.Lemmatize(tweets)
        tweets=self.preprocessor.convert_to_lowercase(tweets)
        return tweets

    def label_tweets(self,tweets, mode='weighted'):

        tweets=self.preprocess_data(tweets)
        if mode=="majority":
           tweets= self.label_tweets_majority(tweets)

        if mode=="weighted":
           tweets= self.label_tweets_weighted(tweets, [0.2, 0.2, 0.2,0.2,0.2],0.5)



        self.evaluate_predictions(tweets['label'],tweets['g_label'])
        y_test=tweets['g_label']
        y_pred=tweets['label']
        print("\nEnsemble ROC-AUC score: %.3f" % roc_auc_score(y_test, tweets['label_prob']))
        print("precison:  %.3f"  %precision_score(y_test, y_pred))
        print("Recall:  %.3f"%recall_score(y_test, y_pred))
        print("F1:  %.3f" %f1_score(y_test, y_pred))

        tweets.to_json(self.filePath, orient='records', lines=True)

    def label_tweets_majority(self, tweets):

        ####  binary label for majority vote #####
        tweets = self.blacklist_detector.predict(tweets,prob=False)
        tweets = self.whitelist_detector.predict(tweets,prob=False)

        x = self.supervised_detector.extract_features(tweets)
        tweets['rf_label'] = self.supervised_detector.predict(self.supervised_detector.rf_model, x,prob=False)
        tweets['lr_label'] = self.supervised_detector.predict(self.supervised_detector.lr_model, x,prob=False)
        tweets['nb_label'] = self.supervised_detector.predict(self.supervised_detector.nb_model, x,prob=False)

        tweets = self.user_desc_detector.predict(tweets,prob=False)
        # tweets = self.cluster_detector.predict(tweets,prob=False)
        tweets['txt_label'] = self.text_detector.predict(tweets, prob=False)

        return self.get_majority_vote(tweets)

    def label_tweets_weighted(self, tweets,weighted,threshold):

        ###  probability label for weighted vote #####
        tweets = self.blacklist_detector.predict(tweets)
        tweets = self.whitelist_detector.predict(tweets)

        x = self.supervised_detector.extract_features(tweets)
        tweets['rf_label'] = self.supervised_detector.predict(self.supervised_detector.rf_model, x)[:,1]
        tweets['lr_label'] = self.supervised_detector.predict(self.supervised_detector.lr_model, x)[:,1]
        tweets['nb_label'] = self.supervised_detector.predict(self.supervised_detector.nb_model, x)[:,1]

        tweets = self.user_desc_detector.predict(tweets)

        # tweets = self.cluster_detector.predict(tweets)
        tweets['txt_label'] = self.text_detector.predict(tweets, prob=False)

        return self.get_ensemble_vote(tweets,weighted,threshold)


    def get_majority_vote(self,tweets,base_learners_result):
        base_learners_result['result'] =base_learners_result[['rf_label', 'ud_label', 'txt_label']].mode(axis=1)[0]
        base_learners_result['result'].loc[base_learners_result.bl_label == 1] = 1
        base_learners_result['result'].loc[base_learners_result.wl_label == 0] = 0
        return base_learners_result['result']
        return tweets

    def get_ensemble_vote(self,tweets,weights,threshhold):
        df = tweets[['lr_label', 'lr_label','rf_label', 'nb_label', 'ud_label', 'cl_label','txt_label']]

        tweets['label_prob'] = df.mean(axis=1)
        tweets['label']=[1 if n>threshhold else 0 for n in tweets['label_prob'] ]
        tweets.loc[(tweets.bl_label == 1), 'label'] = 1
        tweets.loc[(tweets.wl_label == 0), 'label'] = 0

        return tweets

    def get_stacking_vote(self,tweets,base_learners_result):
        # return self.meta_learner.predict_proba(base_learners_result[['bl_label','wl_label','rf_label', 'ud_label', 'txt_label']])[:, 1]

        base_learners_result['result']=self.meta_learner.predict_proba(base_learners_result[['rf_label', 'ud_label', 'txt_label']])[:, 1]
        base_learners_result['result'].loc[base_learners_result.bl_label == 1] = 1
        base_learners_result['result'].loc[base_learners_result.wl_label == 0] = 0
        return base_learners_result['result']

    def get_weighted_average_vote(self,tweets,base_learners_result):
        b = np.array([0.1, 0.1,0.1,0.1,0.6])
        return np.mean(base_learners_result[['bl_label','wl_label','rf_label', 'ud_label', 'txt_label']].to_numpy()*(b), axis=1)

    def get_simple_average_vote(self,tweets,base_learners_result):

        # return np.mean(base_learners_result[['bl_label','wl_label','rf_label', 'ud_label', 'txt_label']].to_numpy(), axis=1)
        base_learners_result['result'] = np.mean(base_learners_result[['rf_label', 'ud_label', 'txt_label']].to_numpy(), axis=1)
        base_learners_result['result'].loc[base_learners_result.bl_label == 1] = 1
        base_learners_result['result'].loc[base_learners_result.wl_label == 0] = 0
        return base_learners_result['result']



    def add_confident_tweets(self,tweets, low_threshod,high_threshold):
        self.confident_tweets.append(tweets[tweets['label_prob'] >=high_threshold])
        self.confident_tweets.append(tweets[tweets['label_prob'] <=low_threshod])

    def evaluate_predictions(self,Y_pred, Y_true):
        fp = 0
        tp = 0
        fn = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == 1:
                if Y_pred[i] == Y_true[i]:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if Y_pred[i] != Y_true[i]:
                 fn = fn + 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = 2 * (precision * recall) / (precision + recall)

        return (precision, recall, fmeasure)

    def predict_base_learners(self, inp, verbose=True):
        """
        Generate a prediction matrix.
        """

        tweets =self.blacklist_detector.predict(inp)
        tweets =self.whitelist_detector.predict(tweets)
        # tweets = self.cluster_detector.predict(tweets)

        x = self.supervised_detector.extract_features(tweets)
        tweets['rf_label'] = self.supervised_detector.predict( self.supervised_detector.rf_model, x)[:, 1]
        # tweets['lr_label'] = self.supervised_detector.predict( self.supervised_detector.lr_model, x)[:, 1]
        # tweets['nb_label'] = pred_base_learners[3].predict(pred_base_learners[3].nb_model, x)[:, 1]
        tweets['txt_label'] = self.text_detector.predict(tweets)
        tweets = self.user_desc_detector.predict(tweets)
        return tweets


    def annotate(self,data,mode="stacking"):
        tweets=self.preprocess_data(data)
        tweets=self.predict_base_learners(tweets)
        base_learners_result= tweets[['bl_label', 'wl_label', 'rf_label', 'ud_label', 'txt_label']]


        if mode=="majority":
           tweets= self.label_tweets_majority(tweets)

        if mode=="average":
            tweets['avg_label_prob'] = self.get_simple_average_vote(tweets, base_learners_result)
            tweets['wavg_label_prob'] = self.get_weighted_average_vote(tweets, base_learners_result)

        if mode=="stacking":
            tweets['label_prob']= self.get_stacking_vote(tweets,base_learners_result)
            tweets['label_prob'] = self.get_stacking_vote(tweets, base_learners_result)


        return  tweets








