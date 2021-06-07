import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone
from tqdm import tqdm
from numpy import savetxt

import matplotlib.pyplot as plt


from src.framework.blacklistdetector import BlackListUserDetector
from src.framework.clusterdetector import ClusterDetector
from src.framework.preprocessor import Preprocessor
from src.framework.semiduperviseddetector import SemiSupervisedDetector
from src.framework.superviseddetector import SupervisedDetector
from src.framework.textdetector import TextDetector
from src.framework.userdescriptiondetector import UserDescriptionDetector
from src.framework.whitelistdetector import WhiteListDetector

SEED = 222
class BatchUpdate():

    def __init__(self, file_path):
        self.file_path=file_path
        self.preprocessor = Preprocessor()




    def load_data(self, filepath):
        try:
            return pd.read_json(filepath, orient='records', lines=True)
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            print('Decoding JSON has failed')
        except FileNotFoundError:
            print('No File')

    def preprocess_data(self,tweets):
        tweets=self.preprocessor.remove_urls(tweets)
        tweets=self.preprocessor.remove_emoji(tweets)
        tweets=self.preprocessor.remove_sign(tweets)
        tweets=self.preprocessor.replace_special_char_abb(tweets)
        tweets=self.preprocessor.Lemmatize(tweets)
        tweets=self.preprocessor.convert_to_lowercase(tweets)
        tweets = self.preprocessor.remove_stop_words(tweets)
        return tweets

    def train_base_learners(self, x_train, y_train, verbose=False,input='batch',save=False):
        """
        Train all base learners in the library.
        """
        file_path = self.file_path+input+'/'
        blacklist_detector = BlackListUserDetector(blackListUser_filePath=file_path + "user_blackList.json",
                                                   userDictionary_filepath=file_path + "user_hijacked_tweets_counts.json",
                                                   threshhold=5)
        whitelist_detector = WhiteListDetector(whiteListUser_filePath=file_path + "user_whiteList.json",
                                               userDictionary_filepath=file_path + "user_vaid_tweets_counts.json",
                                               hijackedDictionary_filepath=file_path + "hijacked_dict.json",
                                               validDictionary_filepath=file_path + "valid_dict.json",
                                               user_threshhold=8, word_threshhold=0.8)
        # cluster_detector = ClusterDetector(file_path + "km_Model.sav", file_path + "vector.pkl",
        #                                    file_path + "cluster_", threshold=0.6)
        supervised_detector = SupervisedDetector(rf_model_filepath=file_path + "rf_Model.sav",
                                                 lr_model_filepath=file_path + "lr_Model.sav",
                                                 nb_model_filepath=file_path + "nb_Model.sav")
        user_desc_detector = UserDescriptionDetector(model_filePath=file_path + "userDesc_count_Model.sav")
        text_detector = TextDetector(model_filePath=file_path + "text_Model.sav")
        blacklist_detector.update_black_List_user(x_train,save=save)
        whitelist_detector.update_white_List_user(x_train,save=save)
        # cluster_detector.fit(x_train,save=save)
        supervised_detector.fit(x_train,save=save)
        user_desc_detector.fit(x_train,save=save)
        text_detector.fit(x_train, save=save)
        return [blacklist_detector,whitelist_detector,supervised_detector,user_desc_detector,text_detector]

    def get_base_learners(self, x_train, y_train, verbose=True,input='batch'):
        """
        Train all base learners in the library.
        """
        file_path = self.file_path+input+'/'
        blacklist_detector = BlackListUserDetector(blackListUser_filePath=file_path+"user_blackList.json",
                                                        userDictionary_filepath=file_path+"user_hijacked_tweets_counts.json",
                                                        threshhold=5)
        whitelist_detector = WhiteListDetector(whiteListUser_filePath=file_path+"user_whiteList.json",
                                                    userDictionary_filepath=file_path+"user_vaid_tweets_counts.json",
                                                    hijackedDictionary_filepath=file_path+"hijacked_dict.json",
                                                    validDictionary_filepath=file_path+"valid_dict.json",
                                                    user_threshhold=8, word_threshhold=0.6)
        # cluster_detector = ClusterDetector(file_path+"km_Model.sav", file_path+"vector.pkl",
        #                                         file_path+"cluster_", threshold=0.6)
        supervised_detector = SupervisedDetector(rf_model_filepath=file_path+"rf_Model.sav",
                                                      lr_model_filepath=file_path+"lr_Model.sav",
                                                      nb_model_filepath=file_path+"nb_Model.sav")
        user_desc_detector = UserDescriptionDetector(model_filePath=file_path+"userDesc_count_Model.sav")
        text_detector = TextDetector(model_filePath=file_path + "text_Model.sav")

        # return [blacklist_detector,whitelist_detector,cluster_detector,supervised_detector,user_desc_detector,text_detector]

        return [blacklist_detector, whitelist_detector, supervised_detector, user_desc_detector,
            text_detector]

    def predict_base_learners(self,pred_base_learners, inp, verbose=True):
        """
        Generate a prediction matrix.
        """

        tweets =pred_base_learners[0].predict(inp)
        tweets = pred_base_learners[1].predict(tweets)
        # tweets = pred_base_learners[2].predict(tweets)

        x = pred_base_learners[2].extract_features(tweets)
        tweets['rf_label'] = pred_base_learners[2].predict( pred_base_learners[2].rf_model, x)[:, 1]
        # tweets['lr_label'] = pred_base_learners[2].predict( pred_base_learners[2].lr_model, x)[:, 1]
        # tweets['nb_label'] = pred_base_learners[3].predict(pred_base_learners[3].nb_model, x)[:, 1]

        tweets = pred_base_learners[3].predict(tweets)
        tweets['txt_label']=pred_base_learners[4].predict(tweets)
        return tweets[['bl_label','wl_label', 'rf_label','ud_label','txt_label','label']]


    def ensemble_predict(self,base_learners, meta_learner, inp, verbose=True):
        """
        Generate predictions from the ensemble.
        """
        P_pred = self.predict_base_learners(base_learners, inp, verbose=verbose)
        P_pred['result'] = meta_learner.predict_proba(P_pred[['rf_label', 'ud_label', 'txt_label']])[:, 1]
        P_pred['result'].loc[P_pred.bl_label == 1] = 1
        P_pred['result'].loc[P_pred.wl_label == 0] = 0
        return P_pred, P_pred['result']




    def get_ensembe(self,training):

        meta_learner = GradientBoostingClassifier(
            n_estimators=1000,
            loss="exponential",
            max_features=3,
            max_depth=3,
            subsample=0.5,
            learning_rate=0.5,
            random_state=SEED)
        # meta_learner=RandomForestClassifier(max_depth=2, random_state=0)
        # meta_learner=KNeighborsClassifier(n_neighbors=3)
        # meta_learner = AdaBoostClassifier(n_estimators=100, random_state=0)
        xtrain_base, xpred_base, ytrain_base, ypred_base = train_test_split(
            training, training["label"], test_size=0.5, random_state=SEED)

        base_learners=self.train_base_learners(xtrain_base, ytrain_base)
        P_base = self.predict_base_learners(base_learners, xpred_base)
        # meta_learner.fit(P_base[['bl_label','wl_label','rf_label', 'ud_label', 'txt_label']], P_base['label'])
        meta_learner.fit(P_base[['rf_label', 'ud_label', 'txt_label']], P_base['label'])
        base_learners = self.train_base_learners(training,  training["label"],input='model')

        return  base_learners,meta_learner

    def stacking(self, meta_learner, X, y, generator):
        """Simple training routine for stacking."""

        # Train final base learners for test time
        print("Fitting final base learners...", end="")
        # base_learners=self.get_base_learners(X, y, verbose=False,input='model')
        base_learners = self.train_base_learners(X, y, verbose=False, input='model',save=True)
        print("done")

        # Generate predictions for training meta learners
        # Outer loop:
        print("Generating cross-validated predictions...")
        # X.set_index('id', inplace=True)
        cv_preds, cv_y = [], []
        for i, (train_idx, test_idx) in enumerate(generator.split(X)):
            fold_xtrain, fold_ytrain = X.loc[train_idx], y[train_idx]
            fold_xtest, fold_ytest = X.loc[test_idx], y[test_idx]

            # Inner loop: step 4 and 5
            # fold_base_learners = {name: clone(model)
            #                       for name, model in base_learners.items()}
            fold_base_learners=self.train_base_learners(
                 fold_xtrain, fold_ytrain, verbose=False)

            fold_P_base = self.predict_base_learners(fold_base_learners,
                 fold_xtest, verbose=False)

            # cv_preds.append(fold_P_base[['bl_label','wl_label','rf_label', 'ud_label', 'txt_label']])
            cv_preds.append(fold_P_base[['rf_label', 'ud_label', 'txt_label']])
            cv_y.append(fold_P_base['label'])
            print("Fold %i done" % (i + 1))

        print("CV-predictions done")

        # Be careful to get rows in the right order
        cv_preds = np.vstack(cv_preds)
        cv_y = np.hstack(cv_y)

        # Train meta learner
        print("Fitting meta learner...", end="")
        meta_learner.fit(cv_preds, cv_y)

        print("done")

        return base_learners, meta_learner

    def evaluate_predictions(self,Y_pred, Y_true, pos_label=1):
        pred = [1 if n > 0.5 else 0 for n in Y_pred]
        precision= precision_score(Y_true,pred, pos_label=pos_label)
        recall= recall_score(Y_true, pred,pos_label=pos_label)
        fmeasure=f1_score(Y_true, pred,pos_label=pos_label)
        # auc_score = auc(recall, precision)
        roc_score = roc_auc_score(Y_true, Y_pred)

        return (roc_score,precision, recall, fmeasure)

    def plot_roc_curve(self, ytest, P_base_learners, P_ensemble, labels, ens_label):
        """Plot the roc curve for base learners and ensemble."""
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--')

        cm = [plt.cm.rainbow(i)
              for i in np.linspace(0, 1.0, P_base_learners.shape[0] + 1)]

        for i in range(P_base_learners.shape[0]):
            p = P_base_learners[i].reshape(-1,1)
            fpr, tpr, _ = roc_curve(ytest, p)

            plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])

        fpr, tpr, _ = roc_curve(ytest, P_ensemble)
        plt.plot(fpr, tpr, label=ens_label, c=cm[0])

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(frameon=False)
        plt.show()

    def plot_precision_recall_curve(self, ytest, P_base_learners, P_ensemble, labels, ens_label):
            """Plot the roc curve for base learners and ensemble."""
            plt.figure(figsize=(10, 8))
            plt.plot([0, 1], [0, 0], 'k--')
            # plot the no skill precision-recall curve


            cm = [plt.cm.rainbow(i)
                  for i in np.linspace(0, 1.0, P_base_learners.shape[0] + 1)]

            for i in range(P_base_learners.shape[0]):
                p = P_base_learners[i].reshape(-1,1)
                pr, rc, _ = precision_recall_curve(ytest, p)

                plt.plot(rc, pr, label=labels[i], c=cm[i + 1])

            pr, rc, _ = precision_recall_curve(ytest, P_ensemble)
            plt.plot(rc, pr, label=ens_label, c=cm[0])

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision_Recall curve')
            plt.legend(frameon=False)
            plt.show()


    def save_model(self,model,file_path):
        pickle.dump(model, open(file_path, 'wb'))

    def update(self,training):
        meta_learner = GradientBoostingClassifier(
            n_estimators=1000,
            loss="exponential",
            max_features=3,
            max_depth=3,
            subsample=0.5,
            learning_rate=0.005,
            random_state=SEED)
        # meta_learner=AdaBoostClassifier(n_estimators=100, random_state=0)

        base_learners, cv_meta_learner = self.stacking(
            clone(meta_learner), training, training['label'], KFold(4))

        self.save_model(cv_meta_learner,self.file_path+"/model/meta_learner.sav")
        return base_learners,cv_meta_learner

def main(args):
    training_path=args[0]
    test_path = args[1]

    batchUpdate= BatchUpdate("../../data/report/")

    training = batchUpdate.load_data(training_path)
    test = batchUpdate.load_data(test_path)
    result = [None] * 100
    bl_result = [None] * 100
    wl_result = [None] * 100
    rf_result = [None] * 100
    ud_result = [None] * 100
    txt_result = [None] * 100
    P_arr = pd.DataFrame()
    P_avg_arr = pd.DataFrame()
    P_mj_arr = pd.DataFrame()

    for i in tqdm(range(0, 100)):
        print("################### %d  Fold #######################" % i)
        meta_learner = GradientBoostingClassifier(
            n_estimators=1000,
            loss="exponential",
            max_features=3,
            max_depth=3,
            subsample=0.5,
            learning_rate=0.005,
            random_state=SEED)

        # meta_learner = AdaBoostClassifier(n_estimators=100, random_state=0)

        base_learners, cv_meta_learner = batchUpdate.stacking(
            clone(meta_learner), training, training['label'], KFold(4))

        P_pred, p = batchUpdate.ensemble_predict(base_learners,cv_meta_learner, test, verbose=False)

        y_test = P_pred['label']
        score= batchUpdate.evaluate_predictions(p, y_test)
        result[i]=score

        row_df = pd.DataFrame([p])
        P_arr = pd.concat([row_df, P_arr])
        # p_mean=P_pred[['bl_label','wl_label', 'rf_label','ud_label','txt_label']].mean(axis=1)
        P_pred['avg-result'] = P_pred[['rf_label','ud_label','txt_label']].mean(axis=1)
        P_pred['avg-result'].loc[P_pred.bl_label == 1] = 1
        P_pred['avg-result'].loc[P_pred.wl_label == 0] = 0
        row_df = pd.DataFrame([P_pred['avg-result']])
        P_avg_arr = pd.concat([row_df, P_avg_arr])
        P_pred['bl_label'].loc[P_pred.bl_label < 1] = 0
        P_pred['wl_label'].loc[P_pred.wl_label > 0] = 1

        P_pred['rf_label'].loc[P_pred.rf_label >0.5] = 1
        P_pred['rf_label'].loc[P_pred.rf_label <= 0.5] = 0
        P_pred['ud_label'].loc[P_pred.ud_label >0.5] = 1
        P_pred['ud_label'].loc[P_pred.ud_label <= 0.5] = 0
        P_pred['txt_label'].loc[P_pred.txt_label >0.5] = 1
        P_pred['txt_label'].loc[P_pred.txt_label <= 0.5] = 0

        score= batchUpdate.evaluate_predictions(P_pred["bl_label"], y_test)
        bl_result[i]=score
        score= batchUpdate.evaluate_predictions(P_pred["wl_label"], y_test,pos_label=0)
        wl_result[i]=score
        score= batchUpdate.evaluate_predictions(P_pred["rf_label"], y_test)
        rf_result[i]=score
        score= batchUpdate.evaluate_predictions(P_pred["ud_label"], y_test)
        ud_result[i]=score
        score= batchUpdate.evaluate_predictions(P_pred["txt_label"], y_test)
        txt_result[i]=score

        P_pred['mj-result'] = P_pred[['rf_label','ud_label','txt_label']].mode(axis=1)[0]
        P_pred['mj-result'].loc[P_pred.bl_label == 1] = 1
        P_pred['mj-result'].loc[P_pred.wl_label == 0] = 0
        row_df = pd.DataFrame([P_pred['mj-result']])
        P_mj_arr = pd.concat([row_df, P_mj_arr])

    score = np.mean(result, axis=0)
    p_mean = np.mean(P_avg_arr.to_numpy(), axis=0)
    p = np.mean(P_arr.to_numpy(), axis=0)
    p_mj = np.mean(P_mj_arr.to_numpy(), axis=0)
    p_bl=np.mean(bl_result, axis=0)
    print('bl Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(p_bl))
    p_wl = np.mean(wl_result, axis=0)
    print('wl Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(p_wl))

    p_rf = np.mean(rf_result, axis=0)
    print('rf Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(p_rf))

    p_ud = np.mean(ud_result, axis=0)
    print('ud Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(p_ud))

    p_txt = np.mean(txt_result, axis=0)
    print('txt Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(p_txt))


    t1=np.array([p_mj, p_mean])
    t2 = np.array([p_mj, p_mean,p_bl,p_wl,p_rf,p_ud,p_txt])

    print('1 Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))
    batchUpdate.plot_roc_curve(y_test, t1, p, ["Majority Vote","Simple Average"], "Meta Learnerer")
    batchUpdate.plot_precision_recall_curve(y_test, t1, p, ["Majority Vote","Simple Average"], "Meta Learnerer")


    df=pd.DataFrame()
    df=df.append({"model":"Meta Learner","roc_auc":score[0],"precision":score[1] ,"recall":score[2],"f1_score":score[3]  },ignore_index=True)

    score= batchUpdate.evaluate_predictions(p_mj, y_test)
    df=df.append({"model":"Majority Vote","roc_auc":score[0],"precision":score[1] ,"recall":score[2],"f1_score":score[3]  },ignore_index=True)

    score= batchUpdate.evaluate_predictions(p_mean, y_test)
    df=df.append({"model":"Simple Average","roc_auc":score[0],"precision":score[1] ,"recall":score[2],"f1_score":score[3]  },ignore_index=True)
    df.to_csv("../../data/report/batch-result.csv")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


