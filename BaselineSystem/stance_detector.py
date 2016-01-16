from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn import svm, cross_validation
from sklearn import dummy
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from nltk.tokenize import word_tokenize
import processTrainingData as ptd
import writePredictionsToFile as write
import os
import numpy as np
import time
import pandas as pd


# ****** TIME **********************************************************************************************************
start_time = time.time()


# ****** CONSTANTS *****************************************************************************************************
TOPIC   = "All"
TOPIC1  = "Atheism"
TOPIC2  = "Climate Change is a Real Concern"
TOPIC3  = "Feminist Movement"
TOPIC4  = "Hillary Clinton"
TOPIC5  = "Legalization of Abortion"


# ****** Settings ******************************************************************************************************
# Topics
topic       = TOPIC2                # Select a topic that will be used as data
test_topic  = TOPIC2                # Select a topic that will be used for testing

use_lemming                 = 0 #nope
# or
use_stemming                = 0 #nope
use_removeAtAndHashtags     = 0 #nope

use_labelprop               = 0
use_svm                     = 1
use_nb                      = 1
use_dummy                   = 1
use_crossval_score          = 0
print_classification_report = 1

# ****** Creating training and test set and preprocess the text ********************************************************
print "\nCreating training set with topic: " + str(topic)
print "Creating test set with topic: " + str(test_topic)
print "\nPreprocessing..."

original_data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
original_targets = list(original_data.Target.unique()) + ['All']

label_prop_data = pd.read_csv(open('label_propagated_data.txt'), '\t', index_col=0)
label_prop_targets = list(original_data.Target.unique()) + ['All']

target_data = original_data[original_data.Target == topic]
print target_data.Tweet

if use_removeAtAndHashtags:
    target_data = ptd.getAllTweetsWithoutHashOrAlphaTag(target_data)
    print target_data.Tweet

if use_lemming:
    target_data = ptd.lemmatizing(target_data)
    print target_data.Tweet
elif use_stemming:
    target_data = ptd.stemming(target_data)
    print target_data.Tweet

if use_labelprop:
    all_data = original_data + label_prop_data

# classifiers = MultinomialNB(), \
#               svm.LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,
#                             intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#                             multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#                             verbose=0), \
#               svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
#                       gamma='auto', kernel='linear', max_iter=-1, probability=True,
#                       random_state=None, shrinking=True, tol=0.001, verbose=False), \
#               dummy.DummyClassifier(strategy='most_frequent', random_state=None, constant=None)


# ****** Features ******************************************************************************************************
print "\nCreating the bag of words..."

# Stop words list can be found at:
#https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/stop_words.py
# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer1Gram = CountVectorizer(analyzer = "word",        # Split the corpus into words
                                  ngram_range = (1,1),      # N-gram: (1,1) = unigram, (2,2) = bigram
                                  stop_words = "english",   # Built-in list of english stop words
                                  decode_error='ignore')

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer2Gram = CountVectorizer(analyzer = "word",        # Split the corpus into words
                                  ngram_range = (2,2),      # N-gram: (1,1) = unigram, (2,2) = bigram
                                  stop_words = "english",   # Built-in list of english stop words.
                                  decode_error='ignore')
# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer3Gram = CountVectorizer(analyzer = "word",        # Split the corpus into words
                                  ngram_range = (3,3),      # N-gram: (1,1) = unigram, (2,2) = bigram
                                  stop_words = "english",   # Built-in list of english stop words.
                                  decode_error='ignore')
vectorizer3GramOptimizedNB = CountVectorizer(analyzer="char",
                                           ngram_range=(3,3),
                                           lowercase=True,
                                           binary=False,
                                           min_df=5,
                                           decode_error='ignore')
vectorizer2_4GramOptimizedSVM = CountVectorizer(analyzer="char_wb",
                                           ngram_range=(2,4),
                                           lowercase=True,
                                           binary=False,
                                           min_df=1,
                                           decode_error='ignore')
vectorizer2_4GramOptimizedSVM2 = CountVectorizer(analyzer="char_wb",
                                           ngram_range=(2,4),
                                           lowercase=False,
                                           binary=False,
                                           min_df=1,
                                           decode_error='ignore')

tfidf_vectorizer1_2OptimizedSVM = TfidfVectorizer(analyzer = 'word',
                                                  ngram_range=(1,2),
                                                  min_df=1)

# ****** Cross validation on classifiers *******************************************************************************
kf = cross_validation.StratifiedKFold(target_data.Stance, n_folds=5, shuffle=True, random_state=1)

# === NAIVE BAYES ===
clf_nb = MultinomialNB()
print clf_nb, '\n'
pipeline_nb = make_pipeline(
    make_union(
        #CountVectorizer(decode_error='ignore'),
        #vectorizer1Gram,
        #vectorizer2Gram,
        #vectorizer3Gram,
        vectorizer3GramOptimizedNB,
        #vectorizer2_4GramOptimizedSVM,
        #vectorizer2_4GramOptimizedSVM2,
        #FunctionTransformer(ptd.determineNegationFeature, validate=False),
        #FunctionTransformer(ptd.lengthOfTweetFeature, validate=False),
        #FunctionTransformer(ptd.numberOfTokensFeature, validate=False),
        #FunctionTransformer(ptd.numberOfCapitalWords, validate=False),
        FunctionTransformer(ptd.numberOfNonSinglePunctMarks, validate=False),
        #FunctionTransformer(ptd.isExclamationMark, validate=False),
        #FunctionTransformer(ptd.numberOfLengtheningWords, validate=False),
        FunctionTransformer(ptd.determineSentiment, validate=False)            # Should be used
    ),
    clf_nb)

pred_stances_nb = cross_validation.cross_val_predict(pipeline_nb, target_data.Tweet,target_data.Stance, cv=kf)

print classification_report(target_data.Stance, pred_stances_nb, digits=4)

macro_f_nb = fbeta_score(target_data.Stance, pred_stances_nb, 1.0,
                         labels=['AGAINST', 'FAVOR'],
                         average='macro')

print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f_nb)


# === LINEAR SVM ===
clf_svm = svm.LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,
                       intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                       multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                       verbose=0)
print clf_svm, '\n'
pipeline_svm = make_pipeline(
    make_union(
        CountVectorizer(decode_error='ignore'),
        #vectorizer1Gram,
        #vectorizer2Gram,
        vectorizer3Gram,
        #vectorizer3GramOptimizedNB,
        vectorizer2_4GramOptimizedSVM,
        #vectorizer2_4GramOptimizedSVM2,
        #tfidf_vectorizer1_2OptimizedSVM,
        #FunctionTransformer(ptd.determineNegationFeature, validate=False),
        #FunctionTransformer(ptd.lengthOfTweetFeature, validate=False),
        FunctionTransformer(ptd.numberOfTokensFeature, validate=False),
        FunctionTransformer(ptd.numberOfCapitalWords, validate=False),
        #FunctionTransformer(ptd.numberOfNonSinglePunctMarks, validate=False),
        #FunctionTransformer(ptd.isExclamationMark, validate=False),
        #FunctionTransformer(ptd.numberOfLengtheningWords, validate=False),
        FunctionTransformer(ptd.determineSentiment, validate=False)
    ),
    clf_svm)

pred_stances_svm = cross_validation.cross_val_predict(pipeline_svm, target_data.Tweet,target_data.Stance, cv=kf)

print classification_report(target_data.Stance, pred_stances_svm, digits=4)

macro_f_svm = fbeta_score(target_data.Stance, pred_stances_svm, 1.0,
                          labels=['AGAINST', 'FAVOR'],
                          average='macro')

print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f_svm)


# # === SVC LINEAR ===
clf_svm2 = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                    gamma='auto', kernel='linear', max_iter=-1, probability=True,
                    random_state=None, shrinking=True, tol=0.001, verbose=False)
print clf_svm2, '\n'
pipeline_svm2 = make_pipeline(
    make_union(
        CountVectorizer(decode_error='ignore'),
        #vectorizer1Gram,
        #vectorizer2Gram,
        #vectorizer3Gram,
        #vectorizer3GramOptimizedNB,
        #vectorizer2_4GramOptimizedSVM,
        vectorizer2_4GramOptimizedSVM2,
        #tfidf_vectorizer1_2OptimizedSVM,
        #FunctionTransformer(ptd.determineNegationFeature, validate=False),
        #FunctionTransformer(ptd.lengthOfTweetFeature, validate=False),
        #FunctionTransformer(ptd.numberOfTokensFeature, validate=False),
        #FunctionTransformer(ptd.numberOfCapitalWords, validate=False),
        #FunctionTransformer(ptd.numberOfNonSinglePunctMarks, validate=False),
        #FunctionTransformer(ptd.isExclamationMark, validate=False),
        #FunctionTransformer(ptd.numberOfLengtheningWords, validate=False),
        #FunctionTransformer(ptd.determineSentiment, validate=False)
    ),
    clf_svm2)

pred_stances_svm2 = cross_validation.cross_val_predict(pipeline_svm2, target_data.Tweet,target_data.Stance, cv=kf)

print classification_report(target_data.Stance, pred_stances_svm2, digits=4)

macro_f_svm2 = fbeta_score(target_data.Stance, pred_stances_svm2, 1.0,
                           labels=['AGAINST', 'FAVOR'],
                           average='macro')

print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f_svm2)


# === DUMMY ===
# clf_dummy = dummy.DummyClassifier(strategy='most_frequent', random_state=None, constant=None)
# print clf_dummy, '\n'
# pipeline_dummy = make_pipeline(
#     make_union(
#         #CountVectorizer(decode_error='ignore'),
#         #vectorizer1Gram,
#         #vectorizer2Gram,
#         #vectorizer3Gram,
#         vectorizer3GramOptimizedNB,
#         #FunctionTransformer(ptd.determineNegationFeature, validate=False),
#         #FunctionTransformer(ptd.lengthOfTweetFeature, validate=False),
#         #FunctionTransformer(ptd.numberOfTokensFeature, validate=False),
#         #FunctionTransformer(ptd.numberOfCapitalWords, validate=False),
#         #FunctionTransformer(ptd.numberOfNonSinglePunctMarks, validate=False),
#         #FunctionTransformer(ptd.isExclamationMark, validate=False),
#         #FunctionTransformer(ptd.numberOfLengtheningWords, validate=False),
#         #FunctionTransformer(ptd.determineSentiment, validate=False)
#     ),
#     clf_dummy)
#
# pred_stances_dummy = cross_validation.cross_val_predict(pipeline_dummy, target_data.Tweet,target_data.Stance, cv=kf)
#
# print classification_report(target_data.Stance, pred_stances_dummy, digits=4)
#
# macro_f_dummy = fbeta_score(target_data.Stance, pred_stances_dummy, 1.0,
#                       labels=['AGAINST', 'FAVOR'],
#                       average='macro')
#
# print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f_dummy)
#
#
# === VOTING ===
vot_clf = VotingClassifier(estimators=[('NB', pipeline_nb),
                                       ('SVM2', pipeline_svm2)],
                           voting='soft',
                           weights=[1, 1])

pred_stances = cross_validation.cross_val_predict(vot_clf, target_data.Tweet, target_data.Stance, cv=kf)
print classification_report(target_data.Stance, pred_stances, digits=4)

macro_f = fbeta_score(target_data.Stance, pred_stances, 1.0,
                      labels=['AGAINST', 'FAVOR'],
                      average='macro')
print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f)

# ****** Printing total time running ***********************************************************************************
print("\n--- %s seconds ---" % (time.time() - start_time))
