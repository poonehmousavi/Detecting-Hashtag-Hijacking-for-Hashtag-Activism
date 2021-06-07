import pandas as pd
import json

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class BlackListUserDetector():
    """
    BlackListUSer detector od semi-supervised module
    """
    def __init__(self, blackListUser_filePath, userDictionary_filepath, threshhold):
        self.blackListUsers= self.load_Dictionary(blackListUser_filePath)
        self.userDictionary=self.load_Dictionary(userDictionary_filepath)
        self.blackListUsers_filepath= blackListUser_filePath
        self.userDictionary_filepath=userDictionary_filepath
        self.threshhold=threshhold



    def load_Dictionary(self,file_path):
        data={}
        try:
            with open(file_path) as json_file:
                data = json.load(json_file)
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            pass
            # print('Decoding JSON has failed')
        except FileNotFoundError:
            # print('No File')
            pass

        return  data


    def load_user_hijackedTweets_count(self,file_path):
        user_hijacked_count = {}

        with open(file_path) as f:
            for line in f:
                (key, val) = line.split()
                user_hijacked_count[key] = val
        return user_hijacked_count


    def predict(self,tweets,prob=True):
        if prob==False:
            tweets['bl_label'] = -1
            for index,tweet in tweets.iterrows():
                if str(tweet['user.id']) in self.blackListUsers.keys():
                    tweets.loc[index, "bl_label"] = 1
            return tweets
        if prob==True:
            tweets['bl_label'] = 0.5
            for index, tweet in tweets.iterrows():
                if tweet['user.id'] in self.blackListUsers.keys():
                    tweets.loc[index, "bl_label"] = 1

            return tweets



###Need to improve oerformance use previous report
    def update_black_List_user(self,tweets,save=False):
        hijacked = tweets[tweets['label'] == 1]
        hijacked = hijacked.groupby('user.id').size().reset_index(name='counts')

        self.userDictionary= hijacked.set_index('user.id')['counts'].to_dict()
        if (save == True):
            self.save_dictionary(self.userDictionary_filepath, self.userDictionary)

        blackList_users =hijacked[hijacked['counts'] >= self.threshhold]
        self.blackListUsers= blackList_users.set_index('user.id')['counts'].to_dict()
        if(save == True):
            self.save_dictionary(self.blackListUsers_filepath,self.blackListUsers)
        return self.blackListUsers



    def save_dictionary(self,filePath,dic):
        with open(filePath, 'w') as fp:
            json.dump(dic, fp, indent=4)



    def check_users(self,grp):
       if grp.size >= self.threshhold:
           self.blackListUsers.add(grp['user.id'])

    def evaluate_predictions(self, Y_pred, Y_true, pos_label=1):
        pred = [1 if n > 0.5 else 0 for n in Y_pred]
        precision = precision_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        recall = recall_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        fmeasure = f1_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        # auc_score = auc(recall, precision)
        roc_score = roc_auc_score(Y_true, Y_pred)

        return (roc_score,precision, recall, fmeasure)









