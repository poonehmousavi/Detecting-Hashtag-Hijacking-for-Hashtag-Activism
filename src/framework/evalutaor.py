from collections import Counter

import numpy as np
import pandas as pd
from numpy import loadtxt
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score, precision_score, recall_score, \
    precision_recall_curve
from mlens.visualization import corrmat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import seaborn as sns


class Evaluator():

    def load_data(self, filepath):
        return pd.read_json(filepath, orient='records', lines=True)

    def evaluate_predictions(self,Y_pred, Y_true, pos_label=1):
        pred = [1 if n > 0.5 else 0 for n in Y_pred]
        precision= precision_score(Y_true,pred, pos_label=pos_label)
        recall= recall_score(Y_true, pred,pos_label=pos_label)
        fmeasure=f1_score(Y_true, pred,pos_label=pos_label)
        # auc_score = auc(recall, precision)
        roc_score = roc_auc_score(Y_true, Y_pred)

        return (roc_score,precision, recall, fmeasure)


    def score_models(self,P, y):
        """Score model in prediction DF"""
        print("Scoring models.")
        for m in P.columns:
            print("%-26s:" % m)
            if(m== 'wl_label'):
                score = self.evaluate_predictions(P.loc[:, m], y,pos_label=0)
            else:
                score = self.evaluate_predictions( P.loc[:, m],y)
            print('roc_auc_score: %.3f \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' %score)
        print("Done.\n")


    def get_correlation_matrix(self,P,ytest):
        plt.figure(figsize=(15, 10))
        matrix = np.triu(P.corr())
        # sns_plot=sns.heatmap(P.corr(), annot=True, mask=matrix)
        sns_plot = sns.heatmap(P.apply(lambda pred: 1 * (pred >= 0.5) - ytest.values).corr(), annot=True, mask=matrix)

        sns_plot.figure.savefig("../../report/finalrun/2/Correlation_Matrix_test4.png")

    def plot_roc_curve(self, ytest, P_base_learners, P_ensemble, labels, ens_label):
        """Plot the roc curve for base learners and ensemble."""
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--')

        cm = [plt.cm.rainbow(i)
              for i in np.linspace(0, 1.0, P_base_learners.shape[0] + 1)]

        for i in range(P_base_learners.shape[0]):
            p = P_base_learners[i].reshape(-1, 1)
            fpr, tpr, _ = roc_curve(ytest, p)

            plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])

        fpr, tpr, _ = roc_curve(ytest, P_ensemble)
        plt.plot(fpr, tpr, label=ens_label, c=cm[0])

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')

        plt.legend(frameon=False,fontsize=14, loc='lower right')
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
            plt.legend(frameon=False, fontsize=12, loc='lower right')
            plt.show()


    def plot_bar(self,df, group):
        # ######  Plot distribution of differnt Categories #############
        df.groupby(group)['id'].nunique().plot(kind='bar', color='#348ABD')
        # plt.savefig(filepath)
        plt.xlabel('Category')
        plt.ylabel('Number of Tweets')
        plt.title("Category Distribution")
        plt.show()

    def create_fleiss_kappa_matrix(self,data):
        y = np.zeros([data.shape[0], 3])
        for index, row in data.iterrows():

            answers = row['answers']
            answers = [answer[0] for answer in answers]
            c = Counter(answers)
            if c.get(-1) != None:
                y[index][0] = c.get(-1)
            if c.get(0) != None:
                y[index][1] = c.get(0)
            if c.get(1) != None:
                y[index][2] = c.get(1)
        return y

    def cohen_kappa_score(self,labeled_data):
        y1 = [answer[0][0] for answer in labeled_data['answers']]
        y2 = [answer[1][0] for answer in labeled_data['answers']]
        score = cohen_kappa_score(y1, y2, labels=None, weights=None, sample_weight=None)
        return score

    def fleiss_kappa(self,data):
        """
        See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
        :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
        :type M: numpy matrix
        """

        M = self.create_fleiss_kappa_matrix(data)
        N, k = M.shape  # N is # of items, k is # of categories
        n_annotators = float(np.sum(M[0, :]))  # # of annotators

        p = np.sum(M, axis=0) / (N * n_annotators)
        P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
        Pbar = np.sum(P) / N
        PbarE = np.sum(p * p)

        kappa = (Pbar - PbarE) / (1 - PbarE)

        return kappa

    def merge_gold_label(self,data,label):
        """
        merge our gold labels with the report
        """
        label = label[["tweet_id", "avg_answer"]]
        label.rename(columns={'tweet_id': 'id'}, inplace=True)
        label.rename(columns={'avg_answer': 'g_label'}, inplace=True)
        data = pd.merge(data, label, on='id', how='inner')

        print("Total Number of tweets: " + str(data.shape[0]))
        print("Number of hard to tell tweets: " + str(data[data['g_label'] == 0].shape[0]))
        print("Number of hijacked tweets: " + str(data[data['g_label'] == -1].shape[0]))
        print("Number of valid tweets: " + str(data[data['g_label'] == 1].shape[0]))

        ###  remove hard to tell report and labeled hijacked as 1 and valid as 0
        hard_to_tell_data = data[data['g_label'] == 0]
        hijacked_data = data[data['g_label'] == -1]
        valid_data = data[data['g_label'] == 1]
        data['g_label'].loc[hard_to_tell_data.index] = -1
        data['g_label'].loc[hijacked_data.index] = 1
        data['g_label'].loc[valid_data.index] = 0

        print("Total Number of tweets: " + str(data.shape[0]))
        print("Number of hard to tell tweets: " + str(data[data['g_label'] == -1].shape[0]))
        print("Number of hijacked tweets: " + str(data[data['g_label'] == 1].shape[0]))
        print("Number of valid tweets: " + str(data[data['g_label'] == 0].shape[0]))
        return data



def main():

    output_path = "../../data/report/final-report/live-result/"
    evaluator=Evaluator()
    all_data=evaluator.load_data(output_path + "TXT_all_live_result-equal-update.json")
    labels = evaluator.load_data("../../data/mturk_live_validation_hits_result.json")
    data=evaluator.merge_gold_label(all_data,labels)
    sample_data=data.query("g_label != -1")
    df = pd.DataFrame()
    score = evaluator.evaluate_predictions(sample_data['label'], sample_data['g_label'])
    # print(' Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))
    df=df.append({'model':'meta-Learner', 'roc-auc':score[0],'precision':score[1],'recall':score[2],'f1_score':score[3] },ignore_index=True)
    print('Meta Learner Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))

    sample_data['bl_label'].loc[sample_data.bl_label < 1] = 0
    score = evaluator.evaluate_predictions(sample_data['bl_label'], sample_data['g_label'])
    df=df.append({'model':'Known Users Classifier_BL', 'roc-auc':score[0],'precision':score[1],'recall':score[2],'f1_score':score[3] },ignore_index=True)

    sample_data['wl_label'].loc[sample_data.wl_label > 0] = 1
    score = evaluator.evaluate_predictions(sample_data['wl_label'], sample_data['g_label'],pos_label=0)
    df=df.append({'model':'Known Users Classifier_WL', 'roc-auc':score[0],'precision':score[1],'recall':score[2],'f1_score':score[3] },ignore_index=True)

    sample_data['rf_label'].loc[sample_data.rf_label >0.5] = 1
    sample_data['rf_label'].loc[sample_data.rf_label <= 0.5] = 0
    score = evaluator.evaluate_predictions(sample_data['rf_label'], sample_data['g_label'])
    df=df.append({'model':'Social Classifier', 'roc-auc':score[0],'precision':score[1],'recall':score[2],'f1_score':score[3] },ignore_index=True)

    sample_data['ud_label'].loc[sample_data.ud_label >0.5] = 1
    sample_data['ud_label'].loc[sample_data.ud_label <= 0.5] = 0
    score = evaluator.evaluate_predictions(sample_data['ud_label'], sample_data['g_label'])
    df=df.append({'model':'User Profile Classifier', 'roc-auc':score[0],'precision':score[1],'recall':score[2],'f1_score':score[3] },ignore_index=True)

    sample_data['txt_label'].loc[sample_data.txt_label >0.5] = 1
    sample_data['txt_label'].loc[sample_data.txt_label <= 0.5] = 0
    score = evaluator.evaluate_predictions(sample_data['txt_label'], sample_data['g_label'])
    df=df.append({'model':'Tweet Text Classifier', 'roc-auc':score[0],'precision':score[1],'recall':score[2],'f1_score':score[3] },ignore_index=True)
    print('TXT Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))

    sample_data['txt_label2'].loc[sample_data.txt_label2 >0.5] = 1
    sample_data['txt_label2'].loc[sample_data.txt_label2 <= 0.5] = 0
    score = evaluator.evaluate_predictions(sample_data['txt_label2'], sample_data['g_label'])
    df=df.append({'model':'STweet Text Classifier-No Update', 'roc-auc':score[0],'precision':score[1],'recall':score[2],'f1_score':score[3] },ignore_index=True)
    print('TXT 2:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))

    sample_data['txt_label3'].loc[sample_data.txt_label3 >0.5] = 1
    sample_data['txt_label3'].loc[sample_data.txt_label3 <= 0.5] = 0
    score = evaluator.evaluate_predictions(sample_data['txt_label3'], sample_data['g_label'])
    df=df.append({'model':'Tweet Text Classifier-Equal Update', 'roc-auc':score[0],'precision':score[1],'recall':score[2],'f1_score':score[3] },ignore_index=True)
    print('TXT 3:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))

    sample_data['txt_label4'].loc[sample_data.txt_label4 >0.5] = 1
    sample_data['txt_label4'].loc[sample_data.txt_label4 <= 0.5] = 0
    score = evaluator.evaluate_predictions(sample_data['txt_label4'], sample_data['g_label'])
    df=df.append({'model':'Tweet Text Classifier-All Update', 'roc-auc':score[0],'precision':score[1],'recall':score[2],'f1_score':score[3] },ignore_index=True)
    print('TXT 4:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))
    #
    score = evaluator.evaluate_predictions(sample_data['topic_label'], sample_data['g_label'])
    df=df.append({'model':'Jain et al.', 'roc-auc':score[0],'precision':score[1],'recall':score[2],'f1_score':score[3] },ignore_index=True)

    score=evaluator.evaluate_predictions(sample_data['avg_label'], sample_data['g_label'])
    print('Simple Average Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))
    df=df.append({'model':'Simple Average', 'roc-auc':score[0],'precision':score[1],'recall':score[2],'f1_score':score[3] },ignore_index=True)
    df.set_index('model')


    # score = evaluator.evaluate_predictions(sample_data['txt_label'], sample_data['g_label'])
    # print('TXT Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))
    # sample_data['bl_label'].loc[sample_data.bl_label < 1] = 0
    # sample_data['wl_label'].loc[sample_data.wl_label > 0] = 1
    # score = evaluator.evaluate_predictions(sample_data['wl_label'], sample_data['g_label'],pos_label=0)
    # print('wl Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))

#     df=pd.DataFrame()
#     score = evaluator.evaluate_predictions(sample_data['bl_label'], sample_data['g_label'])
#     print('wl Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))
#
#     # sample_data['txt_label2'].loc[sample_data.txt_label2 > 0.5] = 1
#     # sample_data['txt_label2'].loc[sample_data.txt_label2 <= 0.5] = 0
#     # score = evaluator.evaluate_predictions(sample_data['txt_label2'], sample_data['g_label'])
#     # print('TXT2 Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))
#
#
#     t = np.array([sample_data['avg_label'],sample_data['txt_label']])
#     evaluator.plot_precision_recall_curve(sample_data['g_label'], t, sample_data["label"], ["Simple Average","Weighted Average","Text Detector"], "Meta Learnerer")
#     evaluator.plot_roc_curve(sample_data['g_label'], t, sample_data["label"], [sample_data['avg_label'],"Weighted Average","Text Detector"], "Meta Learnerer")
#
    d = np.array([sample_data['avg_label'],sample_data['bl_label'], sample_data['wl_label'], sample_data['rf_label'], sample_data['ud_label'], sample_data['txt_label'],sample_data['topic_label']])
    evaluator.plot_precision_recall_curve(sample_data['g_label'], d, sample_data["label"],["Simple Average",'Known Users Classifier_BL', "Known Users Classifier_WL", "Social Classifier","User Profile Classifier", "Tweet Text Classifier","Jain et al."] , "Meta Learnerer")
    evaluator.plot_roc_curve(sample_data['g_label'], d, sample_data["label"], ["Simple Average",'Known Users Classifier_BL', "Known Users Classifier_WL", "Social Classifier","User Profile Classifier", "Tweet Text Classifier","Jain et al."], "Meta Learnerer")
# # # # bl_label', 'wl_label', 'rf_label', 'ud_label', 'txt_label
#

if __name__ == "__main__":
    main()