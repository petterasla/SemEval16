from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import svm, cross_validation
from sklearn import dummy
import processTrainingData as ptd
import writePredictionsToFile as write
import os
import numpy as np

# ****** CONSTANTS *****************************************************************************************************
TOPIC = "All"
TOPIC1 = "Atheism"
TOPIC2 = "Climate Change is a Real Concern"
TOPIC3 = "Feminist Movement"
TOPIC4 = "Hillary Clinton"
TOPIC5 = "Legalization of Abortion"


# ****** Settings ******************************************************************************************************
# Topics
topic = TOPIC2                  # Select a topic that will be used as data
test_topic = TOPIC2             # Select a topic that will be used for testing

# Pre-processing
use_tf_idf = 0                  # 1 = true, 0 = false
use_lemming = 1                 # 1 = true, 0 = false

# Training
use_abstracts = 0               # 1 = true, 0 = false

# Features
use_negation = 1                # 1 = true, 0 = false
use_lengthOfTweet = 1           # 1 = true, 0 = false
use_numberOfTokens = 1          # 1 = true, 0 = false
use_numberOfCapitalWords = 1    # 1 = true, 0 = false
use_numberOfPunctMarks = 0      # 1 = true, 0 = false
use_numberOfLengtheningWord = 0 # 1 = true, 0 = false
use_sentimentAnalyzer = 0       # 1 = true, 0 = false

features_used = use_negation + use_lengthOfTweet + use_numberOfTokens + use_numberOfCapitalWords + use_numberOfPunctMarks + use_numberOfLengtheningWord + use_sentimentAnalyzer

use_trigram = 1                 # 1 = true, 0 = false


# ****** Creating training and test set and preprocess the text ********************************************************
print "Creating training set with topic: " + str(topic)
print "Creating test set with topic: " + str(test_topic)

# Splitting data into train and test data.
#train_data, test_data = ptd.train_test_split_on_stance(ptd.getTopicData(topic), ptd.getTopicData(test_topic), 0.3, 0.3, 0.3)
train_data = ptd.getTopicData(topic)
test_data = ptd.getTopicTestData(topic)

# Getting all the tweets and removing hashtags and @ tags.
train_tweets = ptd.getAllTweetsWithoutHashOrAlphaTag(ptd.getAllTweets(train_data))

# Getting all the hashtags from each tweet
train_hashtags = ptd.getAllHashtags(ptd.getAllTweets(train_data))

# Parsing the hashtags to words, then add them to the tweet again
for index in range(len(train_hashtags)):
    words = ptd.decryptHashtags(train_hashtags[index])
    for word in words:
        train_tweets[index] = train_tweets[index] + " " + word + " "

train_labels = ptd.getAllStances(train_data)
print "train labels:"

# Adding additional data fomr TCP: Against data
if (use_abstracts):
    favor_abs, against_abs = ptd.processAbstracts()
    for abs in against_abs:
        train_tweets.append(abs)
    #for abs in favor_abs:
    #    train_tweets.append(abs)
    for abs in against_abs:
        train_labels.append("AGAINST")
    #for abs in favor_abs:
    #   train_labels.append("FAVOR")

# Lemmatizing the tweets
if (use_lemming):
    train = ptd.lemmatizing(train_tweets)
else:
    train = train_tweets

print "Length of train set and labels should be the same: " + str(len(train)) + " == " + str(len(train_labels))


# ************ Creating test set (not used if cross_validation.train_test_split is used below)************
test_tweets = ptd.getAllTweetsWithoutHashOrAlphaTag(ptd.getAllTweets(test_data))
test_hashtags = ptd.getAllHashtags(ptd.getAllTweets(test_data))
for index in range(len(test_hashtags)):
    words = ptd.decryptHashtags(test_hashtags[index])
    for word in words:
        test_tweets[index] = test_tweets[index] + " " + word + " "

# Lemmatizing
if (use_lemming):
    test = ptd.lemmatizing(test_tweets)
else:
    test = test_tweets
test_labels = ptd.getAllStances(test_data)
print "Length of test set and labels should be the same: " + str(len(test)) + " == " + str(len(test_labels))


# ****** Create a bag of words from the training set ******
print "Creating the bag of words..."

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer1Gram = CountVectorizer(analyzer = "word",         # Split the corpus into words
                                    ngram_range = (1,1),     # N-gram: (1,1) = unigram, (2,2) = bigram
                                    stop_words = "english")  # Built-in list of english stop words.)
# Stop words list can be found at:
#https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/stop_words.py
vectorizer3Gram = CountVectorizer(analyzer = "word",         # Split the corpus into words
                                  ngram_range = (3,3),     # N-gram: (1,1) = unigram, (2,2) = bigram
                                  stop_words = "english")  # Built-in list of english stop words.

# fit_transform() does two functions: First, it fits the model and learns the vocabulary;
# second, it transforms our training data into feature vectors. The input to fit_transform
# should be a list of strings.
if (use_tf_idf):
    print "Applying TF*IDF..."
    # Transforming to a matrix with counted number of words
    count = vectorizer1Gram.fit_transform(train)
    count2 = vectorizer1Gram.transform(test)

    # Creating a TF*IDF transformer
    tfidf_transformer = TfidfTransformer()

    # Transforming the count matrix to the inverse (TD*IDF)
    train_data_features = tfidf_transformer.fit_transform(count)
    test_data_features = tfidf_transformer.transform(count2)

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()
    test_data_features = test_data_features.toarray()
else:
    # Transforming to a matrix with counted number of words
    train_data_features = vectorizer1Gram.fit_transform(train)
    test_data_features = vectorizer1Gram.transform(test)

    freqs = [(word, train_data_features.getcol(idx).sum()) for word, idx in vectorizer1Gram.vocabulary_.items()]
    print sorted (freqs, key = lambda x: -x[1])[:10]

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()
    test_data_features = test_data_features.toarray()

    if use_trigram:
        trigram_train_data_features = vectorizer3Gram.fit_transform(train)
        trigram_test_data_features = vectorizer3Gram.transform(test)

        train_data_features = np.c_[train_data_features, trigram_train_data_features.toarray()]
        test_data_features = np.c_[test_data_features, trigram_test_data_features.toarray()]


# ******* Adding additional features ***********************************************************************************
if features_used > 0:
    print "Addding additional features..."
    if use_negation:
        print "Presens of negation in tweet..."
    if use_lengthOfTweet:
        print "Length of tweet..."
    if use_numberOfTokens:
        print "Number of tokens in tweet..."
    if use_numberOfCapitalWords:
        print "Number of capital words in tweet..."
    if use_numberOfPunctMarks:
        print "Number of punctuation marks in tweet..."
    if use_numberOfLengtheningWord:
        print "Number of lengthened words in tweet..."
    if use_sentimentAnalyzer:
        print "Using sentiment analyzer..."
    trainTable = []
    testTable = []
    for i in range(len(train)):
        # Adding a feature on whether the tweet contains negated segments
        if use_negation:
            trainTable.append([ptd.determineNegationFeature(train[i])])

        if use_lengthOfTweet:
            if len(trainTable) > i:
                trainTable[i].append(ptd.lengthOfTweetFeature(train[i]))
            else:
                trainTable.append([ptd.lengthOfTweetFeature(train[i])])

        if use_numberOfTokens:
            if len(trainTable) > i:
                trainTable[i].append(ptd.numberOfTokensFeature(train[i]))
            else:
                trainTable.append([ptd.numberOfTokensFeature(train[i])])

        if use_numberOfCapitalWords:
            if len(trainTable) > i:
                trainTable[i].append(ptd.numberOfCapitalWords(train[i]))
            else:
                trainTable.append([ptd.numberOfCapitalWords(train[i])])

        if use_numberOfPunctMarks:
            if len(trainTable) > i:
                trainTable[i].append(ptd.numberOfNonSinglePunctMarks(train[i])[0])
                trainTable[i].append(ptd.numberOfNonSinglePunctMarks(train[i])[1])
            else:
                trainTable.append([ptd.numberOfNonSinglePunctMarks(train[i])[0], ptd.numberOfNonSinglePunctMarks(train[i])[1]])

        if use_numberOfLengtheningWord:
            if len(trainTable) > i:
                trainTable[i].append(ptd.numberOfLengtheningWords(train[i]))
            else:
                trainTable.append([ptd.numberOfLengtheningWords(train[i])])

        if use_sentimentAnalyzer:
            sentiments = ptd.determineSentiment(train[i])
            if len(trainTable) > i:
                #trainTable[i].append(sentiments['compound'])
                trainTable[i].append(sentiments['neg'])
                trainTable[i].append(sentiments['neu'])
                trainTable[i].append(sentiments['pos'])
            else:
                #trainTable[i].append(sentiments['compound'])
                trainTable.append([sentiments['neg'], sentiments['neu'], sentiments['pos']])

    for i in range(len(test)):
        # Adding a feature on whether the tweet contains negated segments
        if use_negation:
            testTable.append([ptd.determineNegationFeature(test[i])])

        if use_lengthOfTweet:
            if len(testTable) > i:
                testTable[i].append(ptd.lengthOfTweetFeature(train[i]))
            else:
                testTable.append([ptd.lengthOfTweetFeature(train[i])])

        if use_numberOfTokens:
            if len(testTable) > i:
                testTable[i].append(ptd.numberOfTokensFeature(train[i]))
            else:
                testTable.append([ptd.numberOfTokensFeature(train[i])])

        if use_numberOfCapitalWords:
            if len(testTable) > i:
                testTable[i].append(ptd.numberOfCapitalWords(train[i]))
            else:
                testTable.append([ptd.numberOfCapitalWords(train[i])])

        if use_numberOfPunctMarks:
            if len(testTable) > i:
                testTable[i].append(ptd.numberOfNonSinglePunctMarks(train[i])[0])
                testTable[i].append(ptd.numberOfNonSinglePunctMarks(train[i])[1])
            else:
                testTable.append([ptd.numberOfNonSinglePunctMarks(train[i])[0], ptd.numberOfNonSinglePunctMarks(train[i])[1]])

        if use_numberOfLengtheningWord:
            if len(testTable) > i:
                testTable[i].append(ptd.numberOfLengtheningWords(train[i]))
            else:
                testTable.append([ptd.numberOfLengtheningWords(train[i])])

        if use_sentimentAnalyzer:
            sentiments = ptd.determineSentiment(train[i])
            if len(testTable) > i:
                #testTable[i].append(sentiments['compound'])
                testTable[i].append(sentiments['neg'])
                testTable[i].append(sentiments['neu'])
                testTable[i].append(sentiments['pos'])
            else:
                #testTable.append(sentiments['compound'])
                testTable.append([sentiments['neg'], sentiments['neu'], sentiments['pos']])

    train_data_features = np.c_[train_data_features, trainTable]
    test_data_features = np.c_[test_data_features, testTable]

#print train_data_features


# ******* Cross validation *********************************************************************************************
print "Train classifier using cross validation and SVM..."
clfcv = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                gamma=0.0, kernel='linear', max_iter=-1, probability=True,
                random_state=None, shrinking=True, tol=0.001, verbose=False)

all_features = np.vstack([train_data_features, test_data_features])
all_labels = train_labels + test_labels

"""
kf = cross_validation.StratifiedKFold(all_labels, n_folds=7, shuffle=False)
print "Cross validation scores:"
score = cross_validation.cross_val_score(clfcv, all_features, all_labels, cv=kf, scoring='f1_macro')
print score
print "Cross validation mean:"
print score.mean()
"""

# ******* Train SVM classifier using bag of words **********************************************************************
print "Train classifier using 'train test split' and SVM..."
# Create a SVM model
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
              gamma=0.0, kernel='linear', max_iter=-1, probability=True,
              random_state=None, shrinking=True, tol=0.001, verbose=False)

# Divide into test and train partition
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data_features, train_labels, test_size=0.25, random_state=0)

# Fit model
clf.fit(train_data_features, train_labels)


# ******* Train dummy classifier ***************************************************************************************
print "Train dummy classifier 'train test split'..."
clf_dummy = dummy.DummyClassifier(strategy='most_frequent', random_state=None, constant=None)
clf_dummy.fit(train_data_features, train_labels)


# ******* Predicting test labels ***************************************************************************************
print "Predicting test labels..."
svm_predictions = clf.predict(test_data_features)
dummy_predictions = clf_dummy.predict(test_data_features)
# This is not accurate as the cross validation has already presented the test data to the model in training
#cv_predictions = cross_validation.cross_val_predict(clfcv, test_data_features, test_labels, cv=7)

# ******* Probabilities ************************************************************************************************
# To use the probabilities uncomment the lines 335 to 338 and then comment line 340.
minConfidence = 0.75
svm_predictions_probabilities = clf.predict_proba(test_data_features)

#print svm_predictions_probabilities #against, favor, none


#print 'Score from CV: ' + str(scores)
#print 'Score from test set: ' + str(clf.score(test_data_features, test_labels))


#************ Write to file ********************************************************************************************
print "Writing gold and guesses to file..."
data_file = test_data

# Erwin annotated test data:
annotated = ptd.getAnnotatedData()
erwinsAnnotated = ptd.convertNumberStanceToText([int(row[1]) for row in annotated])

svm_guess_file = write.initFile("guess_svm")
dummy_guess_file = write.initFile("guess_dummy")
#cross_validation_guess_file = write.initFile("guess_cv")
gold_file = write.initFile("gold")
for index in range(len(svm_predictions)):
    #if max(svm_predictions_probabilities[index]) > minConfidence:
    #   write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], svm_predictions[index], svm_guess_file)
    #else:
    #    write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], "NONE", svm_guess_file)

    write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], svm_predictions[index], svm_guess_file)
    write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], dummy_predictions[index], dummy_guess_file)
    #write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], cv_predictions[index], cross_validation_guess_file)
    write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], erwinsAnnotated[index], gold_file)

svm_guess_file.close()
dummy_guess_file.close()
gold_file.close()
#cross_validation_guess_file.close()


#*********** Evaluate the result with the given SemEval16 script *******************************************************
print "\nResults:\n"
print "Dummy prediction score: "
os.system("perl eval.pl gold.txt guess_dummy.txt")
print "SVM prediction score: "
os.system("perl eval.pl gold.txt guess_svm.txt")
#print "SVM cross validation prediction score"
#os.system("perl eval.pl gold.txt guess_cv.txt")