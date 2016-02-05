import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../"))

from glob import glob

import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import FunctionTransformer
import processTrainingData as ptd
from glove_transformer import GloveVectorizer

import BaselineSystem.writePredictionsToFile as write
from numpy import mean


# *****     LOAD DATA     *****
test_data = pd.read_csv(open('SemEval2016-Task6-subtaskA-testdata.txt'), '\t', index_col=0)
#test_data = test_data[test_data.Target == 'Climate Change is a Real Concern']

original_data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
targets = list(original_data.Target.unique())

best_model_score_list = []

for target in targets:
    target_data = original_data[original_data.Target == target]
    print 80 * "="
    print target
    print 80 * "="

    cv = StratifiedKFold(target_data.Stance, n_folds=5, shuffle=True, random_state=1)

    pipeline = Pipeline([
        ('stemming', FunctionTransformer(ptd.stemming, validate=False)),
        ('features', FeatureUnion([

            #('bigram_word', Pipeline([
            #    ('counts', CountVectorizer(decode_error='ignore',
            #                             lowercase=False,
            #                             ngram_range=(2, 2)))
            #])),
            ('trigram_char', Pipeline([
                ('counts', CountVectorizer(analyzer="char",
                                          ngram_range=(3,3),
                                          lowercase=True,
                                          binary=False,
                                          min_df=5,
                                          decode_error='ignore'))
            ])),
            ('neg_feat', Pipeline([
                ('detect', FunctionTransformer(ptd.determineNegationFeature, validate=False))
            ])),
            ('len_feat', Pipeline([
                ('detect', FunctionTransformer(ptd.lengthOfTweetFeature, validate=False))
            ])),
            #('num_of_tokens', Pipeline([
            #    ('detect', FunctionTransformer(ptd.numberOfTokensFeature, validate=False))
            #])),
            ('num_of_cap', Pipeline([
                ('detect', FunctionTransformer(ptd.numberOfCapitalWords, validate=False))
            ])),
            ('num_of_non_single', Pipeline([
                ('detect', FunctionTransformer(ptd.numberOfNonSinglePunctMarks, validate=False))
            ])),
            ('is_exl_mark', Pipeline([
                ('detect', FunctionTransformer(ptd.isExclamationMark, validate=False))
            ])),
            ('num_of_len_words', Pipeline([
                ('detect', FunctionTransformer(ptd.numberOfLengtheningWords, validate=False))
            ])),
            #('sentiment', Pipeline([
            #    ('detect', FunctionTransformer(ptd.determineSentiment, validate=False))
            #]))

        ])),
        ('clf', MultinomialNB())
    ])

    pred_stances = cross_val_predict(pipeline, target_data.Tweet, target_data.Stance, cv=cv)
    print classification_report(target_data.Stance, pred_stances, digits=4)

    macro_f = fbeta_score(target_data.Stance, pred_stances, 1.0,
                          labels=['AGAINST', 'FAVOR'], average='macro')
    print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f)

# Storing the best voting model
    best_model_score_list.append(macro_f)

print best_model_score_list
print mean(best_model_score_list)