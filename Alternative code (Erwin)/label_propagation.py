import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../"))

from glob import glob

import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectFromModel

import BaselineSystem.processTrainingData as ptd
import BaselineSystem.writePredictionsToFile as write
from glove_transformer import GloveVectorizer

from BaselineSystem import cross_val_generator
from BaselineSystem.custom_cross_validation import cross_val_predict   # custom scikit cross validation


from numpy import mean


# *****     LOAD DATA     *****
test_data = pd.read_csv(open('SemEval2016-Task6-subtaskA-testdata.txt'), '\t', index_col=0)
test_data = test_data[test_data.Target == 'Climate Change is a Real Concern']

original_data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
original_data = original_data[original_data.Target == 'Climate Change is a Real Concern']

targets = list(original_data.Target.unique())

label_prop_data = pd.read_csv(open('../BaselineSystem/label_propagated_data.txt'), '\t', index_col=0)

data = original_data


# *****     VARIABLES     *****
best_model_score_list = []
best_model_list = []
best_model_score = 0
glove_fnames = glob('GloVeLabelProp/*.pkl')
glove_ids = [fname.split('/')[-1].split('_')[0] for fname in glove_fnames]


for target in targets:
    include = len(original_data[original_data.Target == target])
    target_data = pd.concat([original_data[original_data.Target == target], label_prop_data[label_prop_data.Target == target]], axis=0)
    #target_data = original_data[original_data.Target == target]
    print 80 * "="
    print target
    print 80 * "="

    # Label prop KFold
    kf = cross_val_generator.generateFolds(target_data.Stance, n_folds=5, shuffle=True,
                                            random_state=1, exclude_from_test=include)

    best_model_score = 0.0
    temp_list = []

    # *****     FINDING BEST VECTOR SPACE     *****
    for fname, glove_id in zip(glove_fnames, glove_ids):
        print 80 * '='
        print 'GLOVE VECTORS:', glove_id
        print 80 * '='

        glove_vecs = pd.read_pickle(fname)

        glove_clf = Pipeline([('features', FeatureUnion([
            ('vect', GloveVectorizer(glove_vecs)),
            #('bigram_word', Pipeline([
            #    ('counts', CountVectorizer(decode_error='ignore',
            #                             lowercase=False,
            #                             ngram_range=(2, 2)))
            #])),
            ('trigram_char', Pipeline([
                ('counts', CountVectorizer(analyzer="char_wb",
                                           ngram_range=(2,4),
                                           lowercase=False,
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
              ('feature_selection', SelectFromModel(LogisticRegression(C=0.1,
                                                                       solver='lbfgs',
                                                                       multi_class='multinomial',
                                                                       class_weight='balanced',
                                                                       ))),
              ('clf', LogisticRegression(C=0.1,
                                         solver='lbfgs',
                                         multi_class='multinomial',
                                         class_weight='balanced',
                                         ))
          ])

        pipeline = Pipeline([
            #('stemming', FunctionTransformer(ptd.stemming, validate=False)),
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
                ('sentiment', Pipeline([
                    ('detect', FunctionTransformer(ptd.determineSentiment, validate=False))
                ]))

            ])),
            #('feature_selection', SelectFromModel(MultinomialNB())),
            ('clf', MultinomialNB())
        ])

        # Creating the voting classifier which will be used
        vot_clf = VotingClassifier(estimators=[('char', pipeline),
                                               ('glove', glove_clf),
                                               ],
                                   voting='hard')

        pred_stances = cross_val_predict(vot_clf, target_data.Tweet, target_data.Stance, cv=kf)
        print classification_report(data.Stance, pred_stances, digits=4)

        macro_f = fbeta_score(data.Stance, pred_stances, 1.0,
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


# Printing the best scores an the mean
print best_model_score_list
print mean(best_model_score_list)

# *****     WRITE PREDICTIONS TO FILE     *****
print "Writing gold and guesses to file on all of the targets..."
unknown_list = []

pred_file = write.initFile("label_prop_prediction")

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

    best_model.fit(target_data.Tweet,target_data.Stance)
    predictions = best_model.predict(test_target_data.Tweet)

    counter_individual = 0
    count_unknowns = 0
    for row  in test_target_data.itertuples():
        id = str(row[0])
        target = str(row.Target)
        tweet = str(row.Tweet)

        write.writePrdictionToFile(id, target, tweet, predictions[counter_individual], pred_file)
        counter_individual += 1
        counter_all += 1
    print "Counter_individual should be equal to individual target: " + str(counter_individual) + " == " + str(len(test_target_data))
    unknown_list.append(count_unknowns)
pred_file.close()

print "counter_all should equal length of total: " + str(counter_all) + " == " + str(len(test_data))

os.system("perl /Users/Henrik/Documents/Datateknikk/Prosjektoppgave/SemEval16/Results/eval.pl /Users/Henrik/Documents/Datateknikk/Prosjektoppgave/SemEval16/Results/climate_gold.txt label_prop_prediction.txt")