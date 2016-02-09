import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../"))

from glob import glob

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import VotingClassifier

from glove_transformer import GloveVectorizer

import BaselineSystem.writePredictionsToFile as write
from numpy import mean
import numpy as np
# *****     LOAD DATA     *****
test_data = pd.read_csv(open('SemEval2016-Task6-subtaskA-testdata.txt'), '\t', index_col=0)
#test_data = test_data[test_data.Target == 'Climate Change is a Real Concern']

original_data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
targets = list(original_data.Target.unique())

# *****     PREPARE VARIABLES     *****
best_model_score_list = []
best_model_list = []
best_weights = [[2,1,3], [3,1,3], [2,2,3],[2,1,3], [3,1,2]]
i = 0
glove_fnames = glob('*.pkl')
glove_ids = [fname.split('/')[-1].split('_')[0] for fname in glove_fnames]

# *****     BUILDING MODEL FOR EACH TARGET     *****
for target in targets:
    target_data = original_data[original_data.Target == target]
    print 80 * "="
    print target
    print 80 * "="

    cv = StratifiedKFold(target_data.Stance, n_folds=5, shuffle=True, random_state=1)

    best_model_score = 0.0
    temp_list = []

    # *****     FINDING BEST VECTOR SPACE     *****
    for fname, glove_id in zip(glove_fnames, glove_ids):
        print 80 * '='
        print 'GLOVE VECTORS:', glove_id
        print 80 * '='

        glove_vecs = pd.read_pickle(fname)

        glove_clf = Pipeline([('vect', GloveVectorizer(glove_vecs)),
                              ('clf', LogisticRegression(C=0.1,
                                                         solver='lbfgs',
                                                         multi_class='multinomial',
                                                         class_weight='balanced',
                                                         ))])

        char_clf = Pipeline([('vect', CountVectorizer(analyzer="char",
                                                      ngram_range=(3,3),
                                                      lowercase=True,
                                                      binary=False,
                                                      min_df=5,
                                                      decode_error='ignore')),
                             ('clf', MultinomialNB())])

        word_clf = Pipeline([('vect', CountVectorizer(decode_error='ignore',
                                                      lowercase=False,
                                                      ngram_range=(2, 2))),
                             ('clf', MultinomialNB())])

        # Creating the voting classifier which will be used
        vot_clf = VotingClassifier(estimators=[('char', char_clf),
                                               ('word', word_clf),
                                               ('glove', glove_clf),
                                               #('clf', clf)
                                               ],
                                   voting='soft',   #Best weights for soft: 3,1,3 - f-score (65.40)
                                   #Best for hard: 1,1,2 - f-score (67.54) for Climate Change..
                                   weights=[best_weights[i][0], best_weights[i][1], best_weights[i][2]])

        pred_stances = cross_val_predict(vot_clf, target_data.Tweet, target_data.Stance, cv=cv)
        print classification_report(target_data.Stance, pred_stances, digits=4)

        macro_f = fbeta_score(target_data.Stance, pred_stances, 1.0,
                              labels=['AGAINST', 'FAVOR'], average='macro')
        print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f)

        # Storng best model so far

        if macro_f > best_model_score:
            best_model = vot_clf
            best_predictions = pred_stances
            best_model_score = macro_f
            temp_list.append(macro_f)


    # Storing the best voting model
    best_model_score_list.append(max(temp_list))
    best_model_list.append(best_model)
    i +=1

print "score list"
print best_model_score_list
print "mean"
avg = mean(best_model_score_list)
print avg
print "std"

var = 0
for res in best_model_score_list:
    var += np.square(res-avg)
print np.sqrt(var)
