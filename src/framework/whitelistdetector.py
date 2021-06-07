import json
from collections import Counter

import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class WhiteListDetector():
    """
    Whitle List detector od semi-supervised module
    """

    def __init__(self, whiteListUser_filePath, userDictionary_filepath, hijackedDictionary_filepath, validDictionary_filepath, user_threshhold, word_threshhold):
        self.whiteListUsers= self.load_Dictionary(whiteListUser_filePath)
        self.userDictionary=self.load_Dictionary(userDictionary_filepath)
        self.whiteListUsers_filepath= whiteListUser_filePath
        self.userDictionary_filepath=userDictionary_filepath
        self.hijackedDictionary=self.load_Dictionary(hijackedDictionary_filepath)
        self.validDictionary = self.load_Dictionary(validDictionary_filepath)
        self.hijackedDictionary_filepath= hijackedDictionary_filepath
        self.validDictionary_filepath=validDictionary_filepath
        self.user_threshhold=user_threshhold
        self.word_threshhold=word_threshhold



    def load_Dictionary(self, file_path):
        data = {}
        try:
            with open(file_path) as json_file:
                data = json.load(json_file)
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            pass
            # print('Decoding JSON has failed')
        except FileNotFoundError:
            pass
            # print('No File')


        return data


    def predict(self, tweets,prob=True):
        if prob ==False:
            tweets['wl_label'] = -1
            tknzr = TweetTokenizer()
            for index, tweet in tweets.iterrows():
                if tweet['bl_label'] != 1 and str(tweet['user.id']) in self.whiteListUsers.keys() :
                    tweets.loc[index, "wl_label"] = 0
                    for word in tknzr.tokenize(tweet['processed_fulltext']):
                        if word in self.hijackedDictionary:
                            tweets.loc[index, "wl_label"] = -1
                            break
            return tweets
        else:
            tweets['wl_label'] = 0.5
            tknzr = TweetTokenizer()
            for index, tweet in tweets.iterrows():
                if tweet['bl_label'] != 1 and tweet['user.id'] in self.whiteListUsers.keys():
                    tweets.loc[index, "wl_label"] = 0
                    for word in tknzr.tokenize(tweet['processed_fulltext']):
                        if word in self.hijackedDictionary:
                            tweets.loc[index, "wl_label"] = 0.5
                            break
            return tweets


    ###Need to improve oerformance use previous report
    ##### Could change how to create valid dictionary using number of users
    def update_white_List_user(self, tweets,save=False):
        valid = tweets[tweets['label'] == 0]
        valid = valid.groupby('user.id').size().reset_index(name='counts')

        self.userDictionary = valid.set_index('user.id')['counts'].to_dict()
        if (save==True):
            self.save_dictionary(self.userDictionary_filepath, self.userDictionary)

        whiteList_users = valid[valid['counts'] >= self.user_threshhold]
        self.whiteListUsers = whiteList_users.set_index('user.id')['counts'].to_dict()
        if (save == True):
            self.save_dictionary(self.whiteListUsers_filepath, self.whiteListUsers)
        self.update_hijacked_dictionary(tweets,save=save)

    def save_dictionary(self, filePath, dic):
        with open(filePath, 'w') as fp:
            json.dump(dic, fp, indent=4)

    def check_users(self, grp):
        # grp_hijacked = grp['label'] == 1

        if grp.size >= self.threshhold:
            self.blackListUsers.add(grp['user.id'])




    def update_hijacked_dictionary(self,data,save=False):
         """
         Update hijacked dictionry in batch update module
         """
         labeled_data= data.loc[data['label'].isin([0,1])]
         total_dictionary, hijacked_dictionaty, valid_dictionary =self.generate_dictionaries(labeled_data)
         i=0
         for word, count  in total_dictionary.items():
             if count==0:
                 print(word)


             n_h=  hijacked_dictionaty[word] if word in hijacked_dictionaty.keys() else 0
             n_v = valid_dictionary[word] if word in valid_dictionary.keys() else 0
             p_h= n_h/count
             p_v=n_v/count
             if p_h > p_v and p_h >= self.word_threshhold:
                 self.hijackedDictionary[word]=count
             elif p_v> p_h and p_v >= self.word_threshhold:
                 self.validDictionary[word] = count
         if (save == True):
             self.save_dictionary(self.hijackedDictionary_filepath,self.hijackedDictionary)
             self.save_dictionary(self.validDictionary_filepath, self.validDictionary)



    def generate_dictionaries(self,data):
        tknzr = TweetTokenizer()
        total_dictionary={}
        hijacked_dictionary={}
        valid_dictionary={}

        max_date=data['created_at'].dt.normalize().max()
        min_date = data['created_at'].dt.normalize().min()
        for index, row in data.iterrows():
            for x in  tknzr.tokenize(row["processed_fulltext"]):
                if len(x) > 1:
                    weight=1 - (max_date-row["created_at"].normalize())/(max_date-min_date)+0.000001
                    if x in total_dictionary:
                        total_dictionary[x]=total_dictionary[x]+weight
                    else:
                        total_dictionary[x] =  weight

                    if row['label'] == 1:
                        if x in hijacked_dictionary:
                            hijacked_dictionary[x] = hijacked_dictionary[x] + weight
                        else:
                            hijacked_dictionary[x] = weight
                    else:
                        if x in valid_dictionary:
                            valid_dictionary[x] = valid_dictionary[x] + weight
                        else:
                            valid_dictionary[x] = weight

        total_dictionary = {k: v for k, v in total_dictionary.items() if v >= 15}
        hijacked_dictionary = {k: v for k, v in hijacked_dictionary.items() if v >= 15}
        valid_dictionary = {k: v for k, v in valid_dictionary.items() if v >= 20}

        return  [total_dictionary,hijacked_dictionary,valid_dictionary]

    def evaluate_predictions(self, Y_pred, Y_true, pos_label=1):
        pred = [1 if n >= 0.5 else 0 for n in Y_pred]
        precision = precision_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        recall = recall_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        fmeasure = f1_score(Y_true, pred, zero_division=0, pos_label=pos_label)
        # auc_score = auc(recall, precision)
        roc_score = roc_auc_score(Y_true, Y_pred)

        return (roc_score,precision, recall, fmeasure)

