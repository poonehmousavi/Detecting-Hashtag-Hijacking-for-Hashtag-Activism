import sys

import pandas as pd
import numpy as np
from collections import defaultdict

# from src.apricot import FeatureBasedSelection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from snorkel.labeling import labeling_function, LabelingFunction
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
Hijacked=1
Valid=0
ABSTAIN=-1


def load_data( filepath):
        return pd.read_json(filepath, orient='records', lines=True)

def get_annotators(hits,threshold):
    annotators_dict=defaultdict(int)

    for index, row in hits.iterrows():
        answers = row['answers']
        if answers == answers:
            for answer in answers:
                annotators_dict[answer[1]] += 1

    annotators_dict = {k: v for k, v in annotators_dict.items() if v > threshold}
    key_max = max(annotators_dict.keys(), key=(lambda k: annotators_dict[k]))
    key_min = min(annotators_dict.keys(), key=(lambda k: annotators_dict[k]))
    print('Number of Annotators: ', len(annotators_dict))
    print('Maximum Value: ', annotators_dict[key_max])
    print('Minimum Value: ', annotators_dict[key_min])
    return annotators_dict

def get_featureSpace(data,dict):

    annotators=set(dict)
    features_columns=set({})
    for annotator in annotators:
        features_columns.add(annotator+'_1')
        features_columns.add(annotator + '_0')
        features_columns.add(annotator + '_-1')
    features_df=pd.DataFrame(0, index=data.index, columns=features_columns)
    for index, row in data.iterrows():
        answers = row['answers']
        for answer in answers:
            if answer[1] in dict:
                label=change_answer(answer[0])
                value=1

                features_df.loc[index, answer[1]+'_'+str(label)]=value
    print('Feature Space Size: ', features_df.shape)
    return features_df


def get_number_of_vote(answers,base=7):
    h_count = 0
    v_count = 0
    ht_count = 0
    for answer in answers:
        if answer[0] == Valid:
            v_count = v_count + 1
        elif answer[0] == Hijacked:
            h_count = h_count + 1
        else:
            ht_count = ht_count + 1
    return (h_count/base, v_count/base, ht_count/base)



def get_annotators_feature(data,dict,path):

    annotators=set(dict)
    features_columns=set({})

    for annotator in annotators:
        features_columns.add(annotator)
    features_df=pd.DataFrame(0, index=data.index, columns=features_columns)
    for index, row in data.iterrows():
        answers = row['answers']
        for answer in answers:
            if answer[1] in dict:
                features_df.loc[index, answer[1]]=1
    print('Feature Space Size: ', features_df.shape)
    features_df.to_json(path, orient='records', lines=True)
    return features_df

def get_label_by_annotators_dic(data,dict):

    annotators=set(dict)
    labels_by_annotators= {}

    for annotator in annotators:
        dic={}
        dic[annotator]={}
        labels_by_annotators[annotator]=dic

    for index, row in data.iterrows():
        answers = row['answers']
        for answer in answers:
            if answer[1] in dict:
                dic=labels_by_annotators[answer[1]]
                dic[row['tweet_id']]=change_answer(answer[0])
                labels_by_annotators[answer[1]]=dic
    print('Feature Space Size: ', len(labels_by_annotators))

    return labels_by_annotators

def change_answer(answer):
    if answer == 1:
        return  0
    elif answer== 0:
        return -1
    elif answer == -1:
        return 1
    else:
        return -1

def keyword_lookup(x, keywords, label):
    if any(word in x.fulltext.lower() for word in keywords):
        return label
    return ABSTAIN


def make_keyword_lf(keywords, label=Hijacked):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )

def get_keyword_lf():
    """Hijacked tweets contain '#food', '#foodporn' """
    keyword_food = make_keyword_lf(keywords=['#food','#foodporn'])

    """Hijacked tweets contain '#tfb','#teamfollowback','#followgain'' """
    keyword_tfb = make_keyword_lf(keywords=['#tfb','#teamfollowback','#followgain'])

    """Hijacked tweets contain #follow4follow', '#followforfollow', '#likeforlike' , '#like4like """
    keyword_follow = make_keyword_lf(keywords=['#follow4follow', '#followforfollow', '#likeforlike' , '#like4like'])

    """Hijacked tweets contain '#apple' , '#iphone' """
    keyword_apple = make_keyword_lf(keywords=['#apple' , '#iphone'])

    """Hijacked tweets contain '#ipad' , '#ipadgames' """
    keyword_ipad = make_keyword_lf(keywords=['#ipad' , '#ipadgames'])

    """Hijacked tweets contain'#ps4live', '#gamer' , '#gaming' ,'#games' , '#gamenight' , '#videogames' """
    keyword_game = make_keyword_lf(keywords=['#ps4live', '#gamer' , '#gaming' ,'#games' , '#gamenight' , '#videogames'])

    """Valid tweets contain''movement','abuse', 'harassment','victim' """
    keyword_women = make_keyword_lf(keywords=['women','movement','abuse', 'harassment','victim','feminism','feminist'],label=Valid)

    return [keyword_food,keyword_follow,keyword_tfb,keyword_ipad,keyword_apple,keyword_game]



def worker_lf(x, worker_dict):
    return worker_dict.get(x.id, ABSTAIN)


def make_worker_lf(worker_id,workers_dicts):
    worker_dict = workers_dicts[worker_id]
    name = f"worker_{worker_id}"
    return LabelingFunction(name, f=worker_lf, resources={"worker_dict": worker_dict})


def train_annotation_TP_Models(train,test):
    trn_annotators_dict = get_annotators(train, 0)
    Y_train=train['avg_answer']
    X_Train=get_featureSpace(train,trn_annotators_dict)
    X_Test = get_featureSpace(test, trn_annotators_dict)

    logreg = LogisticRegression()
    logreg.fit(X_Train, Y_train)
    labels= logreg.predict(X_Test)

    return labels

@labeling_function()
def annotators_trust_lf(x):
    return x.ann_label

def generate_label(df_train,workers_dicts,df_test):
        worker_lfs = [make_worker_lf(worker_id,workers_dicts) for worker_id in workers_dicts]
        key_word_lfs = get_keyword_lf()
        lfs=key_word_lfs + key_word_lfs+[]

        applier = PandasLFApplier(lfs=lfs)
        L_train = applier.apply(df=df_train)
        # L_test = applier.apply(df=df_test)
        Y_test=df_test['label']

        print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())
        majority_model = MajorityLabelVoter()

        # Train LabelModel.
        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train, n_epochs=100, seed=123, log_freq=20, l2=0.1, lr=0.01)

        df_train['g_label']=  label_model.predict(L=L_train)
        # df_train.drop(columns=['ann_label'],inplace=True)
        # majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
        #     "accuracy"
        # ]
        # print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")
        #
        # label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
        #     "accuracy"
        # ]
        # print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
        return  df_train

def main(args):
    training_hit_path=  int(args[0])
    validation_hit_path=  args[1]
    training_hits = load_data(training_hit_path)
    validation_hits = load_data(validation_hit_path)

    training_path = int(args[2])
    validation_path = args[3]
    test_path = int(args[4])



    training = load_data(training_path)
    test = load_data(test_path)
    validation = load_data(validation_path)

    trn_annotators_dict = get_annotators(training_hits, 0)
    labels_by_annotator = get_label_by_annotators_dic(training_hits, trn_annotators_dict)

    training = training[~training['id'].isin(list(validation.id))]
    training['ann_label'] = train_annotation_TP_Models(validation_hits, training_hits)

    training = generate_label(training, labels_by_annotator, test)
    training = training.append(validation)
    print("Total Number of tweets: " + str(training.shape[0]))
    print("Number of hard to tell tweets: " + str(training[training['label'] == -1].shape[0]))
    print("Number of hijacked tweets: " + str(training[training['label'] == 1].shape[0]))
    print("Number of valid tweets: " + str(training[training['label'] == -1].shape[0]))




if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

