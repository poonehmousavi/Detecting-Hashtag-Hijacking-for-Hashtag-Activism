import pandas as pd
import json
import re
import tweepy
import  numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from pandas.io.json import json_normalize

"""
Class for cleaning collected report
"""
ERR_FILE = ".new-report/error.txt"
SPAM_FILE = "new-report/spam.txt"
HAM_FILE = "new-report/ham.txt"
METOO_SPAM_FILE="./MeToo_SPAM_TOTAL.txt"
METOO_Valid_FILE="./Spam/MeToo_SPAM_TOTAL.txt"

OUTPUT_FILE = "spam-category-20171016.txt"

def merge_Data(input_path,category):
    """
    Merge all datas from different spammy hashtags category and assign category labels for them
    """
    tweets_data = []
    with open(input_path, "r") as tweets_file:
        for line in tweets_file:
            try:
                tweet = json.loads(line)
                tweet['category']= category
                tweets_data.append(tweet)
            except:
                continue

    return tweets_data

def loadData(input_path):
    tweets_data = []
    with open(input_path, "r") as tweets_file:
        for line in tweets_file:
            try:
                tweet = json.loads(line)
                tweets_data.append(tweet)
            except:
                continue
    return  tweets_data

def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

def isnan(id):
    if pd.isnull(id):
        return False
    else:
        return True

def filter_duplicates(tweet_datas):
    """
    remove all du;icate , retweet and replies rom initial report set
    get extended fulltxt id necessary , remove unnecessary attributes
    """
    df = pd.DataFrame(json_normalize(tweet_datas),  columns=['created_at', 'id', 'user.id', 'text','extended_tweet.full_text','extended_tweet.entities.hashtags','truncated','retweeted_status.id',
                                                             'in_reply_to_status_id','entities.hashtags', 'entities.urls','extended_tweet.entities.urls',
                                                             'favorite_count','retweet_count','quote_count','user.verified',
                                                             'user.friends_count','user.followers_count','user.created_at','user.description','use.statuses_count','category'])
    df.sort_values("id", inplace=True)
    ###### Removing all duplicate tweets #############
    df= df[df.id != 958845418066120706]
    df.drop_duplicates(subset="id", keep=False, inplace=True)


    ############ Removing all retweets ###############
    spam_df= df[pd.isna(df['retweeted_status.id'])]

    ############ Removing all replies ###############
    spam_df= spam_df[pd.isna(df['in_reply_to_status_id'])]


    ############ Getting Full Text and Hashtags for truncated tweets ###############
    for index,row in spam_df.iterrows():
        if row['truncated']== False:
            spam_df.loc[index,"fulltext"] = row["text"]
            if row['entities.hashtags'] != []:
                spam_df.loc[index, "hashtags"] = [[element['text']] for element in row['entities.hashtags']]
            if row['entities.urls'] != []:
                spam_df.loc[index, "urls"] = [[element['url']] for element in row['entities.urls']]
        else:
            spam_df.loc[index, "fulltext"]=row["extended_tweet.full_text"]
            if row['extended_tweet.entities.hashtags'] != []:
                spam_df.loc[index, "hashtags"] = [[element['text']] for element in row['extended_tweet.entities.hashtags']]

            if row['extended_tweet.entities.urls'] != []:
                spam_df.loc[index, "urls"] = [[element['url']] for element in row['extended_tweet.entities.urls']]


    spam_df["number_of_hashtags"] = [len(x) for x in spam_df['hashtags']]
    spam_df['created_at'] = pd.to_datetime(spam_df.created_at)
    df['YearMonth'] = df['YearMonth'] = pd.to_datetime(df['created_at']).apply(
        lambda x: '{year}-{month}'.format(year=x.year, month=x.month))
    spam_df = spam_df.drop(columns=['text','extended_tweet.full_text','entities.hashtags','extended_tweet.entities.hashtags','truncated','retweeted_status.id', 'in_reply_to_status_id','entities.urls','extended_tweet.entities.urls'])

    return  spam_df


def load_valid_Metoo(file_path):
    tweet_datas = []
    tweet_datas.extend(loadData(file_path+"/metoo-10-15-2017_10-31-2017.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-11-01-2017_11-30-2017.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-12-01-2017_12-30-2017.txt"))

    tweet_datas.extend(loadData(file_path+"/metoo-01-01-2018_01-31-2018.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-02-01-2018_03-01-2018.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-03-01-2018_04-01-2018.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-04-01-2018_05-01-2018.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-05-01-2018_06-01-2018.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-06-01-2018_07-01-2018.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-07-01-2018_08-01-2018.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-08-01-2018_09-01-2018.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-09-01-2018_10-01-2018.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-10-01-2018_11-01-2018.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-11-01-2018_12-01-2018.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-12-01-2018_01-01-2019.txt"))

    tweet_datas.extend(loadData(file_path+"/metoo-01-01-2019_02-01-2019.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-02-01-2019_03-01-2019.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-03-01-2019_04-01-2019.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-04-01-2019_05-01-2019.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-05-01-2019_06-01-2019.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-06-01-2019_07-01-2019.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-07-01-2019_08-01-2019.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-08-01-2019_09-01-2019.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-09-01-2019_10-01-2019.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-10-01-2019_11-01-2019.txt"))
    tweet_datas.extend(loadData(file_path+"/metoo-11-01-2019_11-11-2019.txt"))


    ###### Filter Tweets and save result  #############
    df = filter_duplicates(tweet_datas)
    df.to_csv(file_path +"/valid-metoo-15-10-2017-to-11-11-2019.csv")
    df.to_json(file_path +"/valid-metoo-15-10-2017-to-11-11-2019.json", orient='records', lines=True)
    return df


def load_spam_Metoo(file_path):
    tweet_datas = []
    ###### Merge datas from different Categories and save result  #############
    tweet_datas.extend(merge_Data(file_path+"/metoospam_apple.txt", "Apple"))
    tweet_datas.extend(merge_Data(file_path+"/metoospam_food.txt", "Food"))
    tweet_datas.extend(merge_Data(file_path+"/metoospam_Game.txt", "Game"))
    tweet_datas.extend(merge_Data(file_path+"/metoospam_ipad.txt", "IPad"))
    tweet_datas.extend(merge_Data(file_path+"/metoospam_follow.txt", "Follow"))
    tweet_datas.extend(merge_Data(file_path+"/metoospam_TFB.txt", "TFB"))

    ###### Filter Tweets and save result  #############
    df = filter_duplicates(tweet_datas)
    df.to_csv(file_path +"/spam-metoo-15-10-2017-to-11-11-2019.csv")
    df.to_json(file_path +"/spam-metoo-15-10-2017-to-11-11-2019.json", orient='records', lines=True)
    return  df

def plot_bar(df,group,filepath):
    # ######  Plot distribution of differnt Categories #############
    df.groupby(group)['id'].nunique().plot(kind='bar')
    plt.savefig(filepath)

def split_spam_metoo(df,output_path):
    """
    Split spam tweets into test and training with the same distribution over spammy hashtags category
    """
    sampledf = df.sample(n=100,weights = df.groupby('category')['category'].transform('count'))
    df.drop(sampledf.index, axis=0, inplace=True)
    df.to_csv(output_path+"/training_spam_MeeToo.csv")
    df.to_json(output_path+"/training_spam_MeeToo.json", orient='records', lines=True)
    sampledf.to_csv(output_path+"/test_spam_MeeToo.csv")
    sampledf.to_json(output_path+"/test_spam_MeeToo.json", orient='records', lines=True)
    plot_bar(df, 'category', output_path+'/spam_category_train_distribution.png')
    plot_bar(sampledf,'category',output_path+'/spam_category_test_distribution.png')

def split_ham_metoo(df,output_path):
    """
    Split ham tweets into test and training with the same distribution over month-year
    """
    df['YearMonth'] = df['YearMonth'] = pd.to_datetime(df['created_at']).apply(lambda x: '{year}-{month}'.format(year=x.year, month=x.month))
    test_df = df.sample(n=100,weights = df.groupby('YearMonth')['YearMonth'].transform('count'))
    df.drop(test_df.index, axis=0, inplace=True)
    train_df= df.sample(n=1500,weights = df.groupby('YearMonth')['YearMonth'].transform('count'))
    train_df.to_csv(output_path+"/training_ham_MeeToo.csv")
    train_df.to_json(output_path+"/training_ham_MeeToo.json", orient='records', lines=True)
    test_df.to_csv(output_path+"/test_ham_MeeToo.csv")
    test_df.to_json(output_path+"/test_ham_MeeToo.json", orient='records', lines=True)
    plot_bar(test_df,'created_at',output_path+'/ham_created_date_test_distribution.png')
    plot_bar(train_df, 'created_at',output_path+ '/ham_created_date_train_distribution.png')

def merge_ham_spam(spam_df, ham_df,file_path):
    """
    merge ham and spam tweets
    """
    ham_df=ham_df.drop(columns=['category'])
    spam_df = spam_df.drop(columns=[ 'category'])
    frames = [spam_df,  ham_df]
    result = pd.concat(frames)
    result.sort_values("id", inplace=True)
    result.drop_duplicates(subset="id", keep=False, inplace=True)
    result.to_csv(file_path+".csv")
    result.to_json(file_path + ".json", orient='records', lines=True)
    return result

def drop_y(df):
    # list comprehension of the cols that end with '_y'
    to_drop = [x for x in df if x.endswith('_y')]
    df.drop(to_drop, axis=1, inplace=True)
    return df


def main():
    ###        Merge and PreProcess Ham and Spam Tweets      #######
    load_spam_Metoo("../report/Spam")
    load_valid_Metoo("../report/Ham")

    ###        sample from both Ham and Spam tweets using category and Date distribution      #######
    spam_data = pd.read_json("../report/Spam/spam-metoo-15-10-2017-to-11-11-2019.json",orient='records', lines=True)
    ham_data = pd.read_json("../report/Ham/valid-metoo-15-10-2017-to-11-11-2019.json",orient='records', lines=True)
    split_ham_metoo(ham_data,"../report/Ham")
    split_spam_metoo(spam_data,"../report/Spam")

    ###        merge ham and spam report to generate training and test dataset     #######
    ham_data = pd.read_json("../report/Ham/training_ham_MeeToo.json",orient='records', lines=True)
    spam_data = pd.read_json("../report/Spam/training_spam_MeeToo.json",orient='records', lines=True)
    merge_ham_spam(spam_data,ham_data,'../report/final_data/training')

    ham_data = pd.read_json("../report/Ham/test_ham_MeeToo.json",orient='records', lines=True)
    spam_data = pd.read_json("../report/Spam/test_spam_MeeToo.json",orient='records', lines=True)
    merge_ham_spam(spam_data,ham_data,'../report/final_data/test')




if __name__ == "__main__":
    main()