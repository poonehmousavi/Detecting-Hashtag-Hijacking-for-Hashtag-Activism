import argparse
from logging import Logger

import pandas as pd

from tqdm import tqdm
import numpy as np

from src.collectData.twitterStreamer import TwitterStreamer
from src.framework.batchupdate import BatchUpdate

from src.framework.semiduperviseddetector import SemiSupervisedDetector
from src.framework.twitterStreamer import TwitterStreamer
from src.framework.textdetector import TextDetector


def read_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputpath',  type=str,required=True,
                        help='input file path where all model are saved ')
    parser.add_argument("--outputpath", type=str, required=True,
                        help="outputh file path to save the result")
    parser.add_argument("--live_data_path", type=str, required=True,
                        help="where live report is located")
    parser.add_argument("--training_path", type=str, required=True,
                        help="where initial seed report set is located")
    parser.add_argument("--log_file", type=str, required=True,
                        help="where log file is located")
    parser.add_argument('--stream', type=bool, default=False,
                        help='whether gather report realtime or using live_data_path ')
    parser.add_argument('--update', type=str, default=False,
                        help='updating strategy ')
    return parser



def main():
    parser = read_parameters()
    args = parser.parse_args()
    stream=args.stream
    input_path=args.inputpath
    live_data_path=args.live_data_path
    training_path = args.traing_path
    output_path = args.outputpath
    log_file=args.logfile
    logger = Logger(log_file).create_logger()
    batch_update_mode=args.update
    print("-----------------------------------------------------Starting application------------------------------------------------------")
    if stream==True:
        twitter_streamer = TwitterStreamer(3,logger)
        twitter_streamer.stream_tweets(output_path, ['metoo'])
    else:
        training = pd.read_json(training_path,orient='records', lines=True)
        live_df = pd.read_json(live_data_path, orient='records', lines=True)
        live_df.index = live_df['created_at']
        group_indices = live_df.resample('D').indices
        batchUpdate = BatchUpdate(input_path)
        batchUpdate.update(training)
        semi_supervised_detector = SemiSupervisedDetector(input_path+"model/")

        all=pd.DataFrame()
        weights=pd.DataFrame()
        for t, indices in tqdm(group_indices.items(), total=len(group_indices)):
            df = live_df.iloc[indices]
            result = semi_supervised_detector.annotate(df)
            all=all.append(result,ignore_index=True)
            if result.query("bl_label == 1").shape[0]>0:
                print("bl")
            if result.query("wl_label == 0").shape[0]>0:
                print("wl")

            result.drop(columns=['ud_label', 'bl_label', 'wl_label', 'txt_label','rf_label'],
                        inplace=True)
            hh = result[result['label_prob'] >= 0.7]
            v = result[result['label_prob'] <= 0.3]
            if v.shape[0] > 0 or hh.shape[0]> 0:
                print(pd.datetime.strftime(t, '%Y-%m-%d'))

            if batch_update_mode=="equal":
                if hh.shape[0]> v.shape[0]:
                    new_data=v.append(hh.sample(n=len(v), replace=False), ignore_index=True)
                else:
                    new_data = hh.append(v.sample(n=len(hh), replace=False), ignore_index=True)
                training = training.append(new_data, ignore_index=True)
            elif batch_update_mode=="all":
                training = training.append(v, ignore_index=True)
                training = training.append(hh, ignore_index=True)

            batchUpdate.update(training)
            semi_supervised_detector.reload()
            l=semi_supervised_detector.meta_learner.feature_importances_
            weights=weights.append({ "Social Classifier":l[0],"User Profile Classifier":l[1], "Tweet Text Classifier":l[2]},ignore_index=True)

        training.to_json(output_path + "all_training.json", orient='records', lines=True)
        all.to_json(output_path + "all_live_result.json", orient='records', lines=True)
        weights.to_json(output_path + "all_live-all_-learner-weights.json", orient='records', lines=True)


    print("-----------------------------------------------------Application End------------------------------------------------------")



if __name__ == "__main__":
    main()


