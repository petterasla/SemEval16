import os

from glob import glob

import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
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

import processTrainingData as ptd

use_writeToFile =  1

test_data = pd.read_csv(open('SemEval2016-Task6-subtaskA-testdata.txt'), '\t', index_col=0)
#test_data = test_data[test_data.Target == 'Climate Change is a Real Concern']

data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
data = data[data.Target == 'Climate Change is a Real Concern']
true_stances = data.Stance

cv = StratifiedKFold(true_stances, n_folds=5, shuffle=True, random_state=1)

glove_fnames = glob('*.pkl')
glove_ids = [fname.split('/')[-1].split('_')[0] for fname in glove_fnames]

best_model_score = 0.0

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

    clf = Pipeline([('vect', HashingVectorizer(input='content', encoding='utf-8', decode_error='strict',
                                               strip_accents=None, lowercase=True, preprocessor=None,
                                               tokenizer=None, stop_words=None, token_pattern='(?u)\b\w\w+\b',
                                               ngram_range=(1, 1), analyzer='word', n_features=1048576,
                                               binary=False, norm='l2', non_negative=False)),
                    ('clf', SVC(probability=True))])

    pipeline_svm2 = make_pipeline(
        make_union(
            CountVectorizer(decode_error='ignore'),
            CountVectorizer(analyzer="char_wb",
                            ngram_range=(2,4),
                            lowercase=False,
                            binary=False,
                            min_df=1,
                            decode_error='ignore'),
            HashingVectorizer(input='content', encoding='utf-8', decode_error='strict',
                              strip_accents=None, lowercase=True, preprocessor=None,
                              tokenizer=None, stop_words=None, token_pattern='(?u)\b\w\w+\b',
                              ngram_range=(1, 1), analyzer='word', n_features=1048576,
                              binary=False, norm='l2', non_negative=False)

        ), SVC(probability=True))

    # svm_clf = Pipeline([('char_wb',CountVectorizer(analyzer="char_wb",
    #                                     ngram_range=(2,4),
    #                                     lowercase=False,
    #                                     binary=False,
    #                                     min_df=1,
    #                                     decode_error='ignore')),
    #                     ('clf', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
    #                                 gamma='auto', kernel='linear', max_iter=-1, probability=True,
    #                                 random_state=None, shrinking=True, tol=0.001, verbose=False))])

    vot_clf = VotingClassifier(estimators=[('char', char_clf),
                                           ('word', word_clf),
                                           ('glove', glove_clf)],
                               voting='hard',
                               weights=[1, 1, 2])

    pred_stances = cross_val_predict(vot_clf, data.Tweet, true_stances, cv=cv)
    print classification_report(true_stances, pred_stances, digits=4)

    macro_f = fbeta_score(true_stances, pred_stances, 1.0,
                          labels=['AGAINST', 'FAVOR'], average='macro')
    print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f)

    if macro_f > best_model_score:
        best_model = vot_clf
        best_predictions = pred_stances
        best_model_score = macro_f

if use_writeToFile:
    print "Writing gold and guesses to file..."
    best_model.fit(data.Tweet,true_stances)
    predictions = best_model.predict(test_data.Tweet)

    # Erwin annotated test data:
    annotated = ptd.getAnnotatedData()
    erwinsAnnotated = ptd.convertNumberStanceToText([int(row[1]) for row in annotated])

    print len(erwinsAnnotated) == len(test_data.Stance)

    #svm_guess_file = write.initFile("guess_svm")
    #dummy_guess_file = write.initFile("guess_dummy")
    #nb_guess_file = write.initFile("guess_nb")
    gold_file = write.initFile("gold")
    pred_file = write.initFile("predictions")

    counter = 0
    for row  in test_data.itertuples():
        id = str(row[0])
        target = str(row.Target)
        tweet = str(row.Tweet)

        #if max(svm_predictions_probabilities[index]) > minConfidence:
        #   write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], svm_predictions[index], svm_guess_file)
        #else:
        #   write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], "NONE", svm_guess_file)

        if target != 'Climate Change is a Real Concern':
            write.writePrdictionToFile(id, target, tweet,'UNKNOWN', pred_file)
        else:
            write.writePrdictionToFile(id, target, tweet, predictions[counter], pred_file)
        #write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], nb_predictions[index], nb_guess_file)
        #write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], dummy_predictions[index], dummy_guess_file)
        #write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], data_file[index][3], gold_file)
        #write.writePrdictionToFile(id, target, tweet, erwinsAnnotated[counter], gold_file)
        counter += 1

    #svm_guess_file.close()
    #dummy_guess_file.close()
    gold_file.close()
    #nb_guess_file.close()
    pred_file.close()


    #*********** Evaluate the result with the given SemEval16 script *******************************************************
    print "\nResults:\n"
    #print "Dummy prediction score: "
    #os.system("perl eval.pl gold.txt guess_dummy.txt")
    #print "SVM prediction score: "
    #os.system("perl eval.pl gold.txt guess_svm.txt")
    #print "Naive Bayes prediction score: "
    #os.system("perl eval.pl gold.txt guess_nb.txt")
    print "Prediction score: "
    os.system("perl /Users/Henrik/Documents/Datateknikk/Prosjektoppgave/SemEval16/BaselineSystem/eval.pl gold.txt predictions.txt")

print best_model_score