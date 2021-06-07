import argparse
import sys

import tweepy
import json

import pandas as pd



def read_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputpath',  type=str,required=True,
                        help='dataset to load.')
    parser.add_argument("--outputpath", type=str, required=True,
                        help="input file path for labels (.mat)")
    parser.add_argument("--errorpath", type=str, required=True,
                        help="input file path for embedding (.npy)")
    parser.add_argument('--apinu', type=int, default=1,
                        help='Embedding Dimension')
    return parser



def main(args):
    data = pd.read_json("../../data/Final_Live/Processed/f/Jan_May.json", orient='records', lines=True)
    add_data = pd.read_json("../../data/Final_Live/retweet_count/retweet.json", orient='records', lines=True)

    data['favorite_count']=add_data['favorite_count']
    data['retweet_count'] = add_data['retweet_count']

    data.to_json("../../report/Final_Live/retweet_count/Fina_Live.json", orient='records', lines=True)

    parser = read_parameters()
    args = parser.parse_args()
    input_path=args.inputpath
    output_path = args.outputpath
    error_path=args.errorpath
    api_num=args.apinu
    # # Pooneh account
    if api_num ==1:
        consumer_key = "LRaCG3MCynUxoHyUtw6Zlnu1t"
        consumer_secret = "PUG4Dc22NF7Uwjva0NNMXmGfpWx2kAjpKt2Ix776qx4GCfUimd"
        access_token = "1086487147450982405-2Ij3Av5QpC5InVLguX8DsdLErCt3LY"
        access_token_secret = "xjklgbufN4RAoKlZ8aK980SqqADRmQcahArFivTWiqOhI"
    elif api_num==2:
        consumer_key = "2TWTUrOdmij6Z8MZ5oNmdFjnJ"
        consumer_secret = "rMbO2ULSQ7avFpbgszzq5lIjv3ihkRN2omOPjI7AhRjhLi7OxX"
        access_token = "1175128737462935552-VIHnJcFZZ1y5kXDqUpmW11y9q5Pd76"
        access_token_secret = "FPbPWIkAsypyZFhtNfAsceT28u4Mt6KEWhOMOzTW8HdxJ"
    elif api_num==3:
        consumer_key = "x86BELYSBQPhEsUbJIyLPC8qD"
        consumer_secret = "TQyBEnU2vQFd0SBGlC1cPHw1OEkhm4A41R2OqPSIWWRbrpRx4p"
        access_token = "1192166927243698176-GUwMKUvWXtu4m9m8tcHgzGLHMZU36Y"
        access_token_secret = "s2JWvXwiJl4xubvyu9EpX38qUKWaEepxrPMAf7rRAfGHx"

    # Pass OAuth details to tweepy's OAuth handler
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)


    data=pd.read_json(input_path, orient='records', lines=True)
    data.index=data.id
    suspend = open(error_path, "w")
    api = tweepy.API(auth, wait_on_rate_limit=True)
    id_list = data.id
    df = pd.DataFrame(0, index=data.id, columns=['favorite_count', 'retweet_count'])

    for id in id_list:
        try:
            tweet = api.get_status(id=id)
            df.loc[id, 'favorite_count'] = tweet.favorite_count
            df.loc[id, 'retweet_count'] = tweet.retweet_count

        except tweepy.TweepError as e:
            x = {
                "id": id,
                "user":  str(data.loc[id, 'user.id'])
            }
            suspend.write(json.dumps(x) + '\n')

    suspend.close()
    df.to_json(output_path, orient='records', lines=True)








if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))