import os

from glob import glob

import pandas as pd

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

from glove_transformer import GloveVectorizer

import writePredictionsToFile as write
from numpy import mean
import processTrainingData as ptd

use_writeToFile_climate = 0
use_writeToFile_all = 1


test_data = pd.read_csv(open('SemEval2016-Task6-subtaskA-testdata.txt'), '\t', index_col=0)
#test_data = test_data[test_data.Target == 'Climate Change is a Real Concern']

original_data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
targets = list(original_data.Target.unique())

best_model_score_list = []
best_model_list = []
best_weights = [[2,1,3], [3,1,3], [2,2,3],[2,1,3], [3,1,2]]
i = 0
glove_fnames = glob('*.pkl')
glove_ids = [fname.split('/')[-1].split('_')[0] for fname in glove_fnames]
for target in targets:
    target_data = original_data[original_data.Target == target]
    print 80 * "="
    print target
    print 80 * "="

    cv = StratifiedKFold(target_data.Stance, n_folds=5, shuffle=True, random_state=1)

    best_model_score = 0.0
    temp_list = []

    for fname, glove_id in zip(glove_fnames, glove_ids):
        print 80 * '='
        print 'GLOVE VECTORS:', glove_id
        print 80 * '='

        glove_fname = 'glove/semeval2016-task6-trainingdata_climate_glove.twitter.27B.{}d.pkl'
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

        if macro_f > best_model_score:
            best_model = vot_clf
            best_predictions = pred_stances
            best_model_score = macro_f
            temp_list.append(macro_f)

    best_model_score_list.append(max(temp_list))
    best_model_list.append(best_model)
    i +=1

climate_data = original_data[original_data.Target == "Climate Change is a Real Concern"]
climate_best_f_score = 0.0

for fname, glove_id in zip(glove_fnames, glove_ids):
    print 80 * '='
    print 'GLOVE VECTORS:', glove_id
    print 80 * '='
    print "CLIMATE IS A CONCERN...=?=!?#=!?"
    print 80 * '='

    glove_fname = 'glove/semeval2016-task6-trainingdata_climate_glove.twitter.27B.{}d.pkl'
    glove_vecs = pd.read_pickle(fname)

    cv = StratifiedKFold(climate_data.Stance, n_folds=5, shuffle=True, random_state=1)
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

    vot_clf = VotingClassifier(estimators=[('char', char_clf),
                                           ('word', word_clf),
                                           ('glove', glove_clf),
                                           #('clf', clf)
                                           ],
                               voting='hard',   #Best weights for soft: 3,1,3 - f-score (65.40)
                               #Best for hard: 1,1,2 - f-score (67.54) for Climate Change..
                               weights=[1, 1, 2])

    pred_stances = cross_val_predict(vot_clf, climate_data.Tweet, climate_data.Stance, cv=cv)
    print classification_report(climate_data.Stance, pred_stances, digits=4)

    macro_f = fbeta_score(climate_data.Stance, pred_stances, 1.0,
                          labels=['AGAINST', 'FAVOR'], average='macro')
    print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f)
    if macro_f > climate_best_f_score:
        best_climate_model = vot_clf
        climate_best_f_score = macro_f

print best_model_score_list
print mean(best_model_score_list)

use_threshold = 1
if use_writeToFile_all:
    print "Writing gold and guesses to file on all of the targets..."
    unknown_list = []
    if use_threshold:
        threshold = 0.97
        filename = "predict_all_with_individual_models_and_prob" + str(int(threshold*100))
        pred_file = write.initFile(filename)
    else:
        pred_file = write.initFile("predict_all_with_individual_models")

    counter_all = 0
    counter_against = 0
    for i in range(len(best_model_list)):
        best_model = best_model_list[i]
        print "=" * 80
        print "best model for target: " + targets[i]
        print "=" * 80
        print best_model
        print
        target_data = original_data[original_data.Target == targets[i]]
        test_target_data = test_data[test_data.Target == targets[i]]
        if targets[i] == "Climate Change is a Real Concern":
            best_climate_model.fit(target_data.Tweet, target_data.Stance)
            best_climate_predictions = best_climate_model.predict(test_target_data.Tweet)
            print("Best training climate model...")
        best_model.fit(target_data.Tweet,target_data.Stance)
        predictions = best_model.predict(test_target_data.Tweet)
        if use_threshold:
            probs = best_model.transform(test_target_data.Tweet)

        counter_individual = 0
        count_unknowns = 0
        for row  in test_target_data.itertuples():
            id = str(row[0])
            target = str(row.Target)
            tweet = str(row.Tweet)

            if use_threshold:
                max_prob = max_prob =  max([max(probs[0][counter_individual]), max(probs[1][counter_individual]), max(probs[2][counter_individual])])
                if targets[i] == "Climate Change is a Real Concern" and best_climate_predictions[counter_individual] == "AGAINST":
                    write.writePrdictionToFile(id, target, tweet, best_climate_predictions[counter_individual], pred_file)
                    print("Adding against in climate change..")
                    print best_climate_predictions[counter_individual]
                    counter_against += 1
                elif max_prob > threshold:
                    write.writePrdictionToFile(id, target, tweet, predictions[counter_individual], pred_file)
                else:
                    write.writePrdictionToFile(id, target, tweet, "UNKNOWN", pred_file)
                    count_unknowns += 1
            else:
                write.writePrdictionToFile(id, target, tweet, predictions[counter_individual], pred_file)
            counter_individual += 1
            counter_all += 1
        print "Counter_individual should be equal to individual target: " + str(counter_individual) + " == " + str(len(test_target_data))
        unknown_list.append(count_unknowns)
    pred_file.close()

print "counter_all should equal length of total: " + str(counter_all) + " == " + str(len(test_data))

if use_threshold:
    print
    print "probability threshold: " + str(threshold)
    print "length of test set: " + str(len(test_data))
    print "nr of unkowns:\t" + str(sum(unknown_list))
    print "unknowns:"
    print unknown_list
    print "nr of against: " + str(counter_against)
