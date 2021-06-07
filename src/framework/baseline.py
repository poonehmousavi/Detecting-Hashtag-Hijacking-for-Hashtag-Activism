
from random import randrange

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import  pandas as pd
from string import digits
import nltk
import numpy as np


nltk.download('words')
words = set(nltk.corpus.words.words())

def random_algorithm(training, test):
	output_values = training['label']
	unique = list(set(output_values))
	predicted = list()
	for index, row in test.iterrows():
		index = randrange(len(unique))
		predicted.append(unique[index])
	return predicted

def zero_rule_algorithm_classification(training, test):
	output_values = training['label'].tolist()
	prediction = max(set(output_values), key=output_values.count)
	predicted = [prediction for i in range(test.shape[0])]
	return predicted

def min_rule_algorithm_classification(training, test):
	output_values = training['label'].tolist()
	prediction = min(set(output_values), key=output_values.count)
	predicted = [prediction for i in range(test.shape[0])]
	return predicted

# zero rule algorithm for regression
def zero_rule_algorithm_regression(training, test):
	output_values = training['label'].tolist()
	prediction = sum(output_values) / float(len(output_values))
	predicted = [prediction for i in range(len(test))]
	return predicted

def majority_algorithm(training, test):
	output_values = training['label'].tolist()
	prediction = sum(output_values) / float(len(output_values))
	predicted = [prediction for i in range(len(test))]
	return predicted

def get_corps(output_values):
	# output_values = data['processed_fulltext'].tolist()
	corpus=' '.join(output_values)
	remove_digits = str.maketrans('', '', digits)
	res = corpus.translate(remove_digits)
	result=" ".join(w for w in nltk.wordpunct_tokenize(res)  if w.lower() in words or not w.isalpha())
	return result

def get_top_tf_idf_words(response,feature_names, top_n=2):
    sorted_nzs = np.argsort(response.data)[:-(top_n+1):-1]
    return feature_names[response.indices[sorted_nzs]]





def extract_features(self, data):
	"""
    Create features and normalize them.
    """
	feature_cols = ['user.friends_count', 'user.followers_count', 'retweet_count', 'favorite_count', 'user.verified',
					'number_of_hashtags']
	# Normalize total_bedrooms column

	data = data.loc[:, feature_cols]
	normalized_df = data;
	normalized_df['user.friends_count'] = self.normalize(normalized_df, 'user.friends_count')
	normalized_df['user.followers_count'] = (normalized_df['user.followers_count'] - normalized_df[
		'user.followers_count'].min()) / (normalized_df['user.followers_count'].max() - normalized_df[
		'user.followers_count'].min())
	normalized_df['retweet_count'] = (normalized_df['retweet_count'] - normalized_df['retweet_count'].min()) / (
				normalized_df['retweet_count'].max() - normalized_df['retweet_count'].min())
	normalized_df['favorite_count'] = (normalized_df['favorite_count'] - normalized_df['favorite_count'].min()) / (
				normalized_df['favorite_count'].max() - normalized_df['favorite_count'].min())
	# normalized_df=normalized_df.dropna()
	# Y=self.extract_labels(normalized_df)
	# normalized_df= normalized_df.drop(columns=['id','category'])

	return normalized_df


def normalize(self, data, feature):
	min = data[feature].min()
	max = data[feature].max()
	if min == max:
		return data[feature]
	else:
		return (data[feature] - min) / (
				max - min)


def extract_labels(self, data):
	Y = data['label']
	return Y

def tf_idf_map(metoo,general,test):
   general_corpus=get_corps([str(x) for x in general['processed_fulltext']])
   metoo_corpus = get_corps(metoo['processed_fulltext'].tolist())
   corpus=[metoo_corpus,general_corpus]

   vectorizer = TfidfVectorizer()
   X = vectorizer.fit_transform(corpus)
   vectorizer.get_feature_names()
   feature_array = np.array(vectorizer.get_feature_names())
   tfidf_sorting = np.argsort(X[0].toarray()).flatten()[::-1]
   top_n_metoo = set(feature_array[tfidf_sorting][:10])

   # tfidf_sorting = np.argsort(X[1].toarray()).flatten()[::-1]
   # top_n_general =set(feature_array[tfidf_sorting][:10])
   responses=vectorizer.transform(test["processed_fulltext"])
   predict=[]
   for response in responses:
	   tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
	   top_n = set(feature_array[tfidf_sorting][:10])
	   count=0
	   for word in top_n:
		   if word in top_n_metoo:
			   count=count+1

	   if count >= 5:
		   predict.append(0)
	   else:
		   predict.append(1)

   return predict

def convert_label(x, validation, ):
	if x in set(validation['id']):
		return validation['label']
	else:
		return -1


def evaluate_predictions( Y_pred, Y_true, pos_label=1):
	pred = [1 if n > 0.5 else 0 for n in Y_pred]
	precision = precision_score(Y_true, pred, pos_label=pos_label)
	recall = recall_score(Y_true, pred, pos_label=pos_label)
	fmeasure = f1_score(Y_true, pred, pos_label=pos_label)
	# auc_score = auc(recall, precision)
	roc_score = roc_auc_score(Y_true, Y_pred)

	return (roc_score, precision, recall, fmeasure)


	return (precision, recall, fmeasure)


def main(args):
	training_path = args[0]
	test_path = args[1]
	general_path = args[2]
	general=pd.read_json(general_path, orient='records',
				   lines=True)
	training=pd.read_json(training_path, orient='records',
				   lines=True)
	test=pd.read_json(test_path, orient='records',
				   lines=True)

	Y_predict=tf_idf_map(training,general,test)
	test["topic_label"]=Y_predict
	score=evaluate_predictions(Y_predict,test['label'])
	print('1 Ensemble:roc_auc_score: %.3f  \t Precision: %.3f \t Recall:%.3f \t F-measure: %.3f' % tuple(score))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))