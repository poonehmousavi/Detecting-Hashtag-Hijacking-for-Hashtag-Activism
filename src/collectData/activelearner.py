import os
import sys

import numpy
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm

from  ..framework.textdetector import  TextDetector
from src.framework.superviseddetector import SupervisedDetector
# from src.framework.textdetector import  TextDetector

from src.apricot import FacilityLocationSelection, FeatureBasedSelection
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

import umap

from src.framework.userdescriptiondetector import UserDescriptionDetector


class ActiveLearning:
    """"
    Clustering step of semi-supervised module
    """

    def __init__(self, k, B,Beta, T):
        self.k=k
        self.B=B
        self.T=T
        # self.L=L
        self.Beta=b=Beta

    def load_data(self, filepath):
        return pd.read_json(filepath, orient='records', lines=True)

    def evaluate_predictions(self,Y_pred, Y_true, pos_label=1):
        pred = [1 if n > 0.5 else 0 for n in Y_pred]
        precision= precision_score(Y_true, pred,zero_division=0,pos_label=pos_label)
        recall= recall_score(Y_true, pred,zero_division=0,pos_label=pos_label)
        fmeasure=f1_score(Y_true, pred,zero_division=0,pos_label=pos_label)
        # auc_score = auc(recall, precision)
        roc_score = roc_auc_score(Y_true, Y_pred)

        return (roc_score,precision, recall, fmeasure)

    def learn_txt_classifier(self,data,test,date,count):

        input="model"
        textDetector = TextDetector(model_filePath="../../data/report/%s/rf_Model.sav"%input)
        X_train=data['processed_fulltext']
        Y_train=data['label']
        X_test = test['processed_fulltext']
        Y_test = test['label']



        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

        X_train_v = vectorizer.fit_transform(X_train)


        lr = textDetector.fit_sub(X_train, Y_train)
        y_hat = lr.predict(X_test)
        # acc_FL = (y_hat == Y_test).mean()
        # base_acc = precision_score(Y_test, y_hat, zero_division=0)
        # base_acc = recall_score(Y_test, y_hat, zero_division=0)
        # base_acc = (y_hat == Y_test).mean()
        base_acc = f1_score(Y_test, y_hat, zero_division=0)
        # base_acc = roc_auc_score(Y_test, y_hat)

        base_result = [base_acc] * 9

        X__pairwise = pairwise_distances(X_train_v, metric="euclidean", squared=True)
        model = FacilityLocationSelection(count, verbose=True).fit(X__pairwise)
        X_FL = X_train.iloc[model.ranking]
        y_FL = Y_train.iloc[model.ranking]
        model = FeatureBasedSelection(count).fit(X_train_v)
        X_FS = X_train.iloc[model.ranking]
        y_FS = Y_train.iloc[model.ranking]

        FL_result = []
        FS_result = []
        RND_result = []

        ns = [10, 20, 50, 100, 200, 500, 1000, 2000, count]

        for n in ns:

            lr = textDetector.fit_sub(X_FL.head(n), y_FL.head(n))
            y_hat = lr.predict(X_test)
            # acc_FL = precision_score(Y_test, y_hat, zero_division=0)
            # acc_FL = recall_score(Y_test, y_hat, zero_division=0)
            # acc_FL = (y_hat == Y_test).mean()
            acc_FL = f1_score(Y_test, y_hat, zero_division=0)
            # acc_FL = roc_auc_score(Y_test, y_hat)

            FL_result.append(acc_FL)

            lr = textDetector.fit_sub(X_FS.head(n), y_FS.head(n))
            y_hat = lr.predict(X_test)
            # acc_FS = (y_hat == Y_test).mean()
            # acc_FS = precision_score(Y_test, y_hat, zero_division=0)
            # acc_FS = recall_score(Y_test, y_hat, zero_division=0)
            acc_FS = f1_score(Y_test, y_hat, zero_division=0)
            # acc_FS = roc_auc_score(Y_test, y_hat)
            FS_result.append(acc_FS)

            accs_random_ = []
            for i in range(1, 21):
                try:

                    idxs = numpy.arange(X_train.shape[0])

                    numpy.random.seed(i)
                    numpy.random.shuffle(idxs)

                    X_random = X_train.iloc[idxs]
                    y_random = Y_train.iloc[idxs]

                    lr = textDetector.fit_sub(X_random.head(n), y_random.head(n))
                    y_hat = lr.predict(X_test)
                    # acc_random = (y_hat == Y_test).mean()
                    # acc_random = precision_score(Y_test, y_hat, zero_division=0)
                    # acc_random = recall_score(Y_test, y_hat, zero_division=0)
                    acc_random = f1_score(Y_test, y_hat, zero_division=0)
                    # acc_random = roc_auc_score(Y_test, y_hat)
                    accs_random_.append(acc_random)
                except:
                    print('hi')

            RND_result.append(accs_random_)

            print(n, numpy.mean(accs_random_), acc_FL, acc_FS)

        plt.plot(base_result, color='#228B22')
        plt.plot(FL_result, color='#FF6600')
        plt.plot(FS_result, color='#0000FF')
        plt.plot(numpy.mean(RND_result, axis=1), color='0.3')

        plt.fill_between(range(len(ns)), numpy.min(RND_result, axis=1), numpy.max(RND_result, axis=1),
                         color='0.7')

        plt.scatter(range(len(ns)), base_result, color='#228B22', label="Total Sample")
        plt.scatter(range(len(ns)), FL_result, color='#FF6600', label="Facility Location")
        plt.scatter(range(len(ns)), FS_result, color='#0000FF', label="Feature Selection")
        plt.scatter(range(len(ns)), numpy.mean(RND_result, axis=1), color='0.3', label="Random")
        plt.xticks(range(len(ns)), ns)
        plt.xlim(0, len(ns) - 1)
        plt.ylabel("F1_Score", fontsize=14)
        plt.xlabel("# Examples", fontsize=14)
        plt.legend(fontsize=12, loc='lower right')


        plt.savefig("../../data/report/final-report/live-result/txtclassifier-"+date+".png")
        plt.show()




    def learn_supervised(self,data,test):

        input="test"
        supervised_detector = SupervisedDetector(rf_model_filepath="../../report/finalrun/4/%s/rf_Model.sav"%input,
                                                      lr_model_filepath="../../report/finalrun/4/%s/lr_Model.sav"%input,
                                                      nb_model_filepath="../../report/finalrun/4/%s/nb_Model.sav"%input)
        X_train=supervised_detector.extract_features(data)
        Y_train=supervised_detector.extract_labels(data)
        X_test = supervised_detector.extract_features(test)
        Y_test = test['label']

        lr, rf, nb = supervised_detector.fit_sub(X_train, Y_train, save=False)
        y_hat = lr.predict(X_test)
        # lr_acc = (y_hat == Y_test).mean()
        # lr_acc=precision_score(Y_test, y_hat, zero_division=0)
        # lr_acc = recall_score(Y_test, y_hat, zero_division=0)
        lr_acc=f1_score(Y_test, y_hat,zero_division=0)
        # lr_acc = roc_auc_score(Y_test, y_hat)






        base_lr_result=[lr_acc]*8
        # base_rf_result = [rf_acc] * 8

        model=FacilityLocationSelection(2600, verbose=True).fit(X_train)
        X_FL = X_train.iloc[model.ranking]
        y_FL = Y_train.iloc[model.ranking]
        model = FeatureBasedSelection(2600).fit(X_train.values)
        X_FS = X_train.iloc[model.ranking]
        y_FS = Y_train.iloc[model.ranking]

        ns = [10, 20, 50, 100, 200, 500, 1000, 2000]

        lr_FL_result=[]
        lr_RND_result = []

        FS_result=[]



        for n in ns:
            lr_fl, rf_fl, nb_fl = supervised_detector.fit_sub(X_FL.head(n), y_FL.head(n))
            y_hat = lr_fl.predict(X_test)
            # acc_FL = (y_hat == Y_test).mean()
            # acc_FL=precision_score(Y_test, y_hat, zero_division=0)
            # acc_FL = recall_score(Y_test, y_hat, zero_division=0)
            acc_FL=f1_score(Y_test, y_hat,zero_division=0)
            # acc_FL = roc_auc_score(Y_test, y_hat)
            lr_FL_result.append(acc_FL)

            lr_fl, rf_fl, nb_fl = supervised_detector.fit_sub(X_FS.head(n), y_FS.head(n))
            y_hat = lr_fl.predict(X_test)
            # acc_FS = (y_hat == Y_test).mean()
            # acc_FS=precision_score(Y_test, y_hat, zero_division=0)
            # acc_FS = recall_score(Y_test, y_hat, zero_division=0)
            acc_FS=f1_score(Y_test, y_hat,zero_division=0)
            # acc_FS = roc_auc_score(Y_test, y_hat)

            FS_result.append(acc_FS)


            accs_random_ = []
            for i in range(20):
                idxs = numpy.arange(X_train.shape[0])

                numpy.random.seed(i)
                numpy.random.shuffle(idxs)

                X_random = X_train.iloc[idxs]
                y_random = Y_train.iloc[idxs]

                lr_fl, rf_fl, nb_fl = supervised_detector.fit_sub(X_random.head(n), y_random.head(n))
                y_hat = lr_fl.predict(X_test)
                # acc_random = (y_hat == Y_test).mean()
                # acc_random=precision_score(Y_test, y_hat, zero_division=0)
                # acc_random = recall_score(Y_test, y_hat, zero_division=0)
                acc_random = f1_score(Y_test, y_hat, zero_division=0)
                # acc_random = roc_auc_score(Y_test, y_hat)
                accs_random_.append(acc_random)

            lr_RND_result.append(accs_random_)

            print(n, numpy.mean(accs_random_), acc_FL,acc_FS)

        plt.plot(base_lr_result, color='#228B22')
        plt.plot(lr_FL_result, color='#FF6600')
        plt.plot(FS_result, color='#0000FF')
        plt.plot(numpy.mean(lr_RND_result, axis=1), color='0.3')

        plt.fill_between(range(len(ns)), numpy.min(lr_RND_result, axis=1), numpy.max(lr_RND_result, axis=1), color='0.7')

        plt.scatter(range(len(ns)), base_lr_result, color='#228B22', label="Total Sample")
        plt.scatter(range(len(ns)), lr_FL_result, color='#FF6600', label="Facility Location")
        plt.scatter(range(len(ns)), FS_result, color='#0000FF', label="Feature Selection")
        plt.scatter(range(len(ns)), numpy.mean(lr_RND_result, axis=1), color='0.3', label="Random")
        plt.xticks(range(len(ns)), ns)
        plt.xlim(0, len(ns) - 1)
        plt.ylabel("F1-Score", fontsize=14)
        plt.xlabel("# Examples", fontsize=14)
        plt.legend(fontsize=12, loc='top')

        plt.savefig("fashion-ml.pdf")
        plt.show()





    def learn_user_desc(self,data,test):

        input = "test"
        user_desc_detector = UserDescriptionDetector(
            model_filePath="../../data/model/userDesc_count_Model.sav")
        with_desc = data[pd.notna(data['user.description'])]

        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

        X_train_v = vectorizer.fit_transform(with_desc['user.description'])
        X_train=with_desc['user.description']
        Y_train=with_desc['label']
        with_desc_test=test[pd.notna(test['user.description'])]
        X_test = with_desc_test['user.description']
        Y_test = with_desc_test['label']

        ud = user_desc_detector.fit_sub(with_desc['user.description'], Y_train)
        y_hat = ud.predict(X_test)
        # acc_FL = (y_hat == Y_test).mean()
        # base_acc = precision_score(Y_test, y_hat, zero_division=0)
        # base_acc = recall_score(Y_test, y_hat, zero_division=0)
        # base_acc = (y_hat == Y_test).mean()
        base_acc = f1_score(Y_test, y_hat, zero_division=0)
        # base_acc = roc_auc_score(Y_test, y_hat)

        base_result=[base_acc]*9

        X__pairwise = pairwise_distances(X_train_v, metric="euclidean", squared=True)
        model=FacilityLocationSelection(2500, verbose=True).fit(X__pairwise)
        X_FL = X_train.iloc[model.ranking]
        y_FL = Y_train.iloc[model.ranking]
        model = FeatureBasedSelection(2500).fit(X_train_v)
        X_FS = X_train.iloc[model.ranking]
        y_FS = Y_train.iloc[model.ranking]

        FL_result=[]
        FS_result=[]
        RND_result = []

        ns = [10, 20, 50, 100, 200, 500, 1000, 2000,2500]

        for n in ns:

            ud = user_desc_detector.fit_sub(X_FL.head(n), y_FL.head(n))
            y_hat = ud.predict(X_test)
            # acc_FL = precision_score(Y_test, y_hat, zero_division=0)
            # acc_FL = recall_score(Y_test, y_hat, zero_division=0)
            # acc_FL = (y_hat == Y_test).mean()
            acc_FL = f1_score(Y_test, y_hat, zero_division=0)
            # acc_FL = roc_auc_score(Y_test, y_hat)

            FL_result.append(acc_FL)

            ud = user_desc_detector.fit_sub(X_FS.head(n), y_FS.head(n))
            y_hat = ud.predict(X_test)
            # acc_FS = (y_hat == Y_test).mean()
            # acc_FS = precision_score(Y_test, y_hat, zero_division=0)
            # acc_FS = recall_score(Y_test, y_hat, zero_division=0)
            acc_FS = f1_score(Y_test, y_hat, zero_division=0)
            # acc_FS = roc_auc_score(Y_test, y_hat)
            FS_result.append(acc_FS)

            accs_random_ = []
            for i in range(1,21):
                try:

                    idxs = numpy.arange(X_train.shape[0])

                    numpy.random.seed(i)
                    numpy.random.shuffle(idxs)

                    X_random = X_train.iloc[idxs]
                    y_random = Y_train.iloc[idxs]

                    ud = user_desc_detector.fit_sub(X_random.head(n), y_random.head(n))
                    y_hat = ud.predict(X_test)
                    # acc_random = (y_hat == Y_test).mean()
                    # acc_random = precision_score(Y_test, y_hat, zero_division=0)
                    # acc_random = recall_score(Y_test, y_hat, zero_division=0)
                    acc_random = f1_score(Y_test, y_hat, zero_division=0)
                    # acc_random = roc_auc_score(Y_test, y_hat)
                    accs_random_.append(acc_random)
                except:
                    print('hi')

            RND_result.append(accs_random_)

            print(n, numpy.mean(accs_random_), acc_FL, acc_FS)

        plt.plot(base_result, color='#228B22')
        plt.plot(FL_result, color='#FF6600')
        plt.plot(FS_result, color='#0000FF')
        plt.plot(numpy.mean(RND_result, axis=1), color='0.3')

        plt.fill_between(range(len(ns)), numpy.min(RND_result, axis=1), numpy.max(RND_result, axis=1),
                         color='0.7')

        plt.scatter(range(len(ns)), base_result, color='#228B22', label="Total Sample")
        plt.scatter(range(len(ns)), FL_result, color='#FF6600', label="Facility Location")
        plt.scatter(range(len(ns)), FS_result, color='#0000FF', label="Feature Selection")
        plt.scatter(range(len(ns)), numpy.mean(RND_result, axis=1), color='0.3', label="Random")
        plt.xticks(range(len(ns)), ns)
        plt.xlim(0, len(ns) - 1)
        plt.ylabel("F1_Score", fontsize=14)
        plt.xlabel("# Examples", fontsize=14)
        plt.legend(fontsize=12, loc='top')

        plt.savefig("fashion-ml.pdf")
        plt.show()



    def learn_clustering(self, data, test):
        input = "test"
        cluster_detector = ClusterDetector("../../data/model/km_Model.sav",
                                           "../../data/model/vector.pkl",
                                           "../../data/model/cluster_" , threshold=0.6)
        vectorizer, X_train_v = cluster_detector.get_tfid(data)


        cluster_detector.fit(data)
        result = cluster_detector.predict(test)
        y_hat = [1 if n > 0.5 else 0 for n in result['cl_label']]
        # base_acc = (y_hat == result['g_label']).mean()
        # base_acc = precision_score(result['g_label'], y_hat, zero_division=0)
        # base_acc = recall_score(result['g_label'], y_hat, zero_division=0)
        base_acc = f1_score(result['label'], y_hat, zero_division=0)
        # base_acc = roc_auc_score(result['label'], y_hat)
        print(base_acc)

        base_result=[base_acc]*9

        X__pairwise = pairwise_distances(X_train_v, metric="euclidean", squared=True)
        model=FacilityLocationSelection(2500, verbose=True).fit(X__pairwise)
        data_FL = data.iloc[model.ranking]

        model = FeatureBasedSelection(2500).fit(X_train_v)
        data_FS = data.iloc[model.ranking]


        FL_result=[]
        FS_result=[]
        RND_result = []

        ns = [13, 20, 50, 100, 200, 500, 1000, 2000,2500]

        for n in ns:

            cluster_detector.fit(data_FL.head(n))
            result = cluster_detector.predict(test)
            y_hat = [1 if n > 0.5 else 0 for n in result['cl_label']]
            # acc_FL = (y_hat == result['g_label']).mean()
            # acc_FL = precision_score(result['g_label'], y_hat, zero_division=0)
            # acc_FL = recall_score(result['g_label'], y_hat, zero_division=0)
            acc_FL = f1_score(result['label'], y_hat, zero_division=0)
            # acc_FL = roc_auc_score(result['label'], y_hat)
            FL_result.append(acc_FL)

            cluster_detector.fit(data_FS.head(n))
            result = cluster_detector.predict(test)
            y_hat = [1 if n > 0.5 else 0 for n in result['cl_label']]
            # acc_FS = (y_hat == result['g_label']).mean()
            # acc_FS = precision_score(result['g_label'], y_hat, zero_division=0)
            # acc_FS = recall_score(result['g_label'], y_hat, zero_division=0)
            acc_FS = f1_score(result['label'], y_hat, zero_division=0)
            # acc_FS = roc_auc_score(result['g_label'], y_hat)
            FS_result.append(acc_FS)

            accs_random_ = []
            for i in range(1,21):
                try:

                    idxs = numpy.arange(data.shape[0])

                    numpy.random.seed(i)
                    numpy.random.shuffle(idxs)

                    data_random = data.iloc[idxs]

                    cluster_detector.fit(data_random.head(n))
                    result = cluster_detector.predict(test)
                    y_hat = [1 if n > 0.5 else 0 for n in result['cl_label']]
                    # acc_random = (y_hat == result['g_label']).mean()
                    # acc_random = precision_score(result['g_label'], y_hat, zero_division=0)
                    # acc_random = recall_score(result['g_label'], y_hat, zero_division=0)
                    acc_random = f1_score(result['label'], y_hat, zero_division=0)
                    # acc_random = roc_auc_score(result['g_label'], y_hat)
                    accs_random_.append(acc_random)
                except:
                    print('hi')

            RND_result.append(accs_random_)

            print(n, numpy.mean(accs_random_), acc_FL, acc_FS)

        plt.plot(base_result, color='#228B22')
        plt.plot(FL_result, color='#FF6600')
        plt.plot(FS_result, color='#0000FF')
        plt.plot(numpy.mean(RND_result, axis=1), color='0.3')

        plt.fill_between(range(len(ns)), numpy.min(RND_result, axis=1), numpy.max(RND_result, axis=1),
                         color='0.7')

        plt.scatter(range(len(ns)), base_result, color='#228B22', label="Total Sample")
        plt.scatter(range(len(ns)), FL_result, color='#FF6600', label="Facility Location")
        plt.scatter(range(len(ns)), FS_result, color='#0000FF', label="Feature Selection")
        plt.scatter(range(len(ns)), numpy.mean(RND_result, axis=1), color='0.3', label="Random")
        plt.xticks(range(len(ns)), ns)
        plt.xlim(0, len(ns) - 1)
        plt.ylabel("F1-Score", fontsize=14)
        plt.xlabel("# Examples", fontsize=14)
        plt.legend(fontsize=12, loc='lower right')

        plt.savefig("fashion-ml.pdf")
        plt.show()

    def active_learng(self,training,test):
        pass

















def main(args):
    training = args[0]
    test= args[1]


    activeLearner = ActiveLearning(1000,100,200,10)
    training = activeLearner.load_data(training)

    activeLearner.learn_txt_classifier(training,test)
    activeLearner.learn_user_desc(training,test)
    activeLearner.learn_clustering(training, test)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))




