from sklearn import grid_search
import os

from glob import glob

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import Normalizer
from sklearn import grid_search
from glove_transformer import GloveVectorizer
from sklearn.linear_model import SGDClassifier
from time import time
import writePredictionsToFile as write
from numpy import mean
import processTrainingData as ptd


original_data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
climate_data = original_data[original_data.Target == "Climate Change is a Real Concern"]

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
    #('clf', LinearSVC()),
    #('clf', SGDClassifier())
])

parameters = {
    'vect__analyzer': ('word', 'char', 'char_wb'),
    'vect__min_df': (1, 2, 3, 5),
    'vect__stop_words':('english', None),
    'vect__lowercase':(True, False),
    'vect__ngram_range': ((1, 1), (1, 2), (2,2), (3,3)),
    'tfidf__use_idf': (True, False),
    #'clf__penalty': ('l2'),
    #'clf__loss': ('hinge', 'squared_hinge'),
    #'clf__C':(0.1, 0.5, 1.0, 2.0)
    'clf__alpha':(0.2,0.5,1.0,2)
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time()
    grid_search.fit(climate_data.Tweet, climate_data.Stance)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))