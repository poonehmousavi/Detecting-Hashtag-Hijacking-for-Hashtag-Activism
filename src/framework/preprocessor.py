import string

import nltk,sys
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('wordnet')
import numpy as np,pandas as pd
import nltk.data
from nltk.stem.porter import *
pd.options.mode.chained_assignment = None



class Preprocessor():
    def load_data(self,filepath):
        return pd.read_json(filepath, orient='records', lines=True)

    def remove_urls(self, data):
        """
        Remove URL
        """
        data['processed_fulltext'] = data['fulltext'].map(lambda x: re.sub(r"(?:\@|http(s)??\://)\S+", " ", x))
        # print("-------Remove URL--------")
        # print(tweets['processed_TweetText'])
        return data

    def remove_stop_words(self,data):
        """
        stop word removal
        """
        from nltk.corpus import stopwords
        stop_words = stopwords.words("english")
        newStopWords = ["https", "metoo", "com", "amp"]
        punc = ['.', ',', '"', "'", '?', '/', '\\', '!', ':', ';', '(', ')', '[', ']', '{', '}', "%"]
        stop_words.extend(newStopWords)
        stop_words.extend(punc)
        tknzr = nltk.TweetTokenizer()
        data['processed_fulltext'] = data['processed_fulltext'].apply(lambda x: ' '.join(
            [word for word in tknzr.tokenize(x) if word not in (stop_words) and word not in string.punctuation]))
        # print("-------Remove Stop Word--------")
        # print(tweets['processed_TweetText'])

        return data



    def remove_sign(self,data):
        """
        Remove # and @ sign
        """
        spilthash = re.compile(r"[#@](\w+)")
        data['processed_fulltext'] = data['processed_fulltext'].map(lambda x: re.sub(spilthash, r"\g<1> '", x))
        # print("-------Remove # sign --------")
        # print(tweets['processed_TweetText'])
        return data

    def remove_emoji(self, data):
        """
        Remove EMOJI
        """
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u'\U00010000-\U0010ffff'
                                   u"\u200d"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\u3030"
                                   u"\ufe0f"
                                   "]+", flags=re.UNICODE)

        data['processed_fulltext'] = data['processed_fulltext'].map(lambda x: re.sub(emoji_pattern, " ", x))

        # print("-------Remove emoji--------")
        # print(tweets['processed_TweetText'])
        return data

    def replace_special_char_abb(self,data):
        """
        replace special char and replace abbreviations
        """
        data['processed_fulltext'] = data['processed_fulltext'].apply(lambda x: self.translator(x))
        # print("------ replace special char and replace abbreviations ----")
        # print(tweets['processed_TweetText'])

        return data

    def translator(self,user_string):
        """
         Code From: https://medium.com/nerd-stuff/python-script-to-turn-text-message-abbreviations-into-actual-phrases-d5db6f489222
        """
        import csv, re
        user_string = user_string.split(" ")
        j = 0
        for _str in user_string:
            # File path which consists of Abbreviations.
            fileName = "./slang.txt"

            # File Access mode [Read Mode]
            with open(fileName, "r") as myCSVfile:
                # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
                dataFromFile = csv.reader(myCSVfile, delimiter="=")
                # Removing Special Characters.
                _str = re.sub('[^a-zA-Z0-9]+', '', _str)
                for row in dataFromFile:
                    # Check if selected word matches short forms[LHS] in text file.
                    if _str.upper() == row[0]:
                        # If match found replace it with its appropriate phrase in text file.
                        user_string[j] = row[1]
                myCSVfile.close()
            j = j + 1
        return ' '.join(user_string)

    def stem(self,data):
        """
        stemming
        """
        ps = PorterStemmer()
        data['processed_fulltext'] = data['processed_TweetText'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split() ]))
        # print("------ stemming ----")
        # print(report['processed_fulltext'])
        return data

    def Lemmatize(self,data):
        """
        Lemmatization
        """
        from nltk.stem.wordnet import WordNetLemmatizer
        lmtzr = WordNetLemmatizer()
        data['processed_fulltext'] = data['processed_fulltext'].apply(
            lambda x: ' '.join([lmtzr.lemmatize(word, 'v') for word in x.split()]))
        # print("----------Lemmatization--------------")
        # print(report['processed_fulltext'])
        return data

    def convert_to_lowercase(self,data):
        """
        convert all words into lower case
        """
        data['processed_fulltext'] = data['processed_fulltext'].apply(
            lambda x: ' '.join([word.lower() for word in x.split()]))

        data['processed_fulltext'] = data['processed_fulltext'].apply(lambda x: x.strip())
        # print("-------------lower case --------------")
        # print(tweets['processed_fulltext'])
        return  data

    def merge_gold_label(self,data,label):
        """
        merge our gold labels with the report
        """
        label = label[["tweet_id", "avg_answer"]]
        label.rename(columns={'tweet_id': 'id'}, inplace=True)
        label.rename(columns={'avg_answer': 'label'}, inplace=True)
        data = pd.merge(data, label, on='id', how='inner')

        print("Total Number of tweets: " + str(data.shape[0]))
        print("Number of hard to tell tweets: " + str(data[data['label'] == 0].shape[0]))
        print("Number of hijacked tweets: " + str(data[data['label'] == -1].shape[0]))
        print("Number of valid tweets: " + str(data[data['label'] == 1].shape[0]))

        ###  remove hard to tell report and labeled hijacked as 1 and valid as 0
        hard_to_tell_data = data[data['label'] == 0]
        hijacked_data = data[data['label'] == -1]
        valid_data = data[data['label'] == 1]
        data['label'].loc[hard_to_tell_data.index] = -1
        data['label'].loc[hijacked_data.index] = 1
        data['label'].loc[valid_data.index] = 0

        print("Total Number of tweets: " + str(data.shape[0]))
        print("Number of hard to tell tweets: " + str(data[data['label'] == -1].shape[0]))
        print("Number of hijacked tweets: " + str(data[data['label'] == 1].shape[0]))
        print("Number of valid tweets: " + str(data[data['label'] == 0].shape[0]))
        return data

    def get_conflicts_tweets(self,label_data):
        conflict=pd.DataFrame(columns=label_data.columns)
        non_conflict = pd.DataFrame(columns=label_data.columns)
        for index, row in label_data.iterrows():
            answers = row['answers']
            if  answers[0][0]!=answers[1][0]:
                conflict= conflict.append(row)
            else:
                non_conflict = non_conflict.append(row)


        return conflict,non_conflict

    def get_conflicts(self, noisy_data,clean_data):
        noisy_data = noisy_data[["id", "label"]]
        noisy_data.rename(columns={'label': 'n_label'}, inplace=True)
        data = pd.merge(clean_data, noisy_data, on='id', how='left')
        conflicts=data[data.label != data.n_label]
        print("Number of conflicts is "+str(conflicts.shape[0]))
        return  conflicts

    def get_average_answer(self,labeled_data):
        for index, row in labeled_data.iterrows():
            h_count=0
            v_count=0
            ht_count=0
            for answer in row['answers']:
                if answer[0] == 1:
                    v_count = v_count+1
                elif answer[0] == -1:
                    h_count =h_count+1
                else:
                    ht_count=ht_count+1

            m= max(h_count,v_count,ht_count)
            if v_count==m:
                labeled_data.loc[index, "avg_answer"] = 1
            elif h_count==m:
                labeled_data.loc[index, "avg_answer"] = -1
            else:
                labeled_data.loc[index, "avg_answer"] = 0
        return labeled_data

    def preprocess_data(self,tweets):
        tweets=self.remove_urls(tweets)
        tweets=self.remove_emoji(tweets)
        tweets=self.remove_sign(tweets)
        tweets=self.replace_special_char_abb(tweets)
        tweets=self.Lemmatize(tweets)
        tweets=self.convert_to_lowercase(tweets)
        tweets=self.remove_stop_words(tweets)
        return tweets

def main():
    preprocessor = Preprocessor()
    spam = pd.read_json('../../data/spam-metoo-15-10-2017-to-11-11-2019.json', orient='records',
                        lines=True)
    valid = pd.read_json('../../data/valid-metoo-15-10-2017-to-11-11-2019.json', orient='records',
                        lines=True)

    test = pd.read_json('../../data/noisy_test.json', orient='records',
                           lines=True)
    data=spam.append(valid)
    ids=np.array(test['id'])
    data = data[~data['id'].isin(ids)]
    # data = pd.read_json('../../data/all_general.json', orient='records',
    #                        lines=True)
    tweets = preprocessor.preprocess_data(data)
    tweets.to_json('../../data/processed_all_metoo.json', orient='records',
                           lines=True)

if __name__ == "__main__":
        main()







