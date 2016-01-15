from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import svm, cross_validation
from sklearn import dummy
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score
from nltk.tokenize import word_tokenize
from sklearn.ensemble import VotingClassifier
import processTrainingData as ptd
import writePredictionsToFile as write
import os
import numpy as np
import time


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

# Pre-processing
use_tf_idf                  = 0     # 1 = true, 0 = false
use_lemming                 = 1     # 1 = true, 0 = false

# Classifier
use_svm                     = 1     # 1 = true, 0 = false
use_nb                      = 1     # 1 = true, 0 = false   # - Cant have negative features (like use_sentimentAnalyzer)
use_dummy                   = 1     # 1 = true, 0 = false

# Training
use_abstracts               = 0     # 1 = true, 0 = false
use_skeptical_tweets        = 1     # 1 = true, 0 = false
use_labelprop               = 0     # 1 = true, 0 = false
use_test_train_split        = 0     # 1 = true, 0 = false
use_bigram                  = 0     # 1 = true, 0 = false
use_trigram                 = 1     # 1 = true, 0 = false

# Features
use_negation                = 1     # 1 = true, 0 = false   # Returns 0/1 if there exists negated words in the tweet
use_lengthOfTweet           = 1     # 1 = true, 0 = false   # Returns length of the tweet x/140
use_numberOfTokens          = 1     # 1 = true, 0 = false   # Returns number of words in the tweet
use_numberOfCapitalWords    = 1     # 1 = true, 0 = false   # Returns number of capital words in the tweet
use_numberOfPunctMarks      = 0     # 1 = true, 0 = false   # Returns number of non-single punct. marks in tweet(i.e !!)
use_numberOfLengtheningWord = 1     # 1 = true, 0 = false   # Returns number of words that are lengthen (i.e: cooool)
use_sentimentAnalyzer       = 1     # 1 = true, 0 = false   # Returns number between -1 and 1 as compound of pos,neu,neg
use_posAndNegWord           = 1     # 1 = true, 0 = false   # Returns a list [pos, neg] based on number of pos/neg words
# WARNING - BENEATH TAKES 4EVER (20 min med kun topic=climate)
use_numberOfPronouns        = 0     # 1 = true, 0 = false   # Returns number of pronouns in the tweet

features_used = use_negation + use_lengthOfTweet + use_numberOfTokens + use_numberOfCapitalWords + use_numberOfPunctMarks \
                + use_numberOfLengtheningWord + use_sentimentAnalyzer + use_posAndNegWord + use_numberOfPronouns

# Write prediction to file
use_writeToFile             = 0     # 1 = true, 0 = false

# Print classification report based on cross validation training
use_crossval_score          = 0     # 1 = true, 0 = false
print_classification_report = 1     # 1 = true, 0 = false


# ****** Creating training and test set and preprocess the text ********************************************************
print "\nCreating training set with topic: " + str(topic)
print "Creating test set with topic: " + str(test_topic)
print "\nPreprocessing..."

if use_labelprop:
    label_prop_data = ptd.getLabelPropTopicData(topic)
    original_data = ptd.getTopicData(topic)

    train_data = label_prop_data + original_data
    # Test data is not used when not test train split is beeing used
    # but to not change too much code it is just beeing set to two samples
    # should be set to use the semeval test data when system is ready for it
    test_data = ptd.getTopicTestData(topic)
    #test_data = original_data[:2]
else:
    if use_test_train_split:
        # Custom splitting of data into train and test data.
        #train_data, test_data = ptd.train_test_split_on_stance(ptd.getTopicData(topic), ptd.getTopicData(test_topic), 0.3, 0.4, 0.3)

        # Standard random splitting of data into train and test data.
        data = ptd.getTopicData(topic)
        train_data, test_data, y_train, y_test = cross_validation.train_test_split(data, ptd.getAllStances(data), test_size=0.33, random_state=33)
    else:
        # Run with training and test data as provided by SemEval.
        train_data = ptd.getTopicData(topic)
        # Test data is not used when not test train split is beeing used
        # but to not change too much code it is just beeing set to two samples
        # should be set to use the semeval test data when system is ready for it
        test_data = ptd.getTopicTestData(topic)
        #test_data = train_data[:2]

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

#****************** Adding additional data ***************************************************
# Data from TCP
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

# Data from skeptical tweets from TCP
if use_skeptical_tweets:
    number_of_tweets = 40      # A total of 155 tweets.
    against_data = ptd.getSkepticalTweets()[:number_of_tweets]
    train_tweets = train_tweets + against_data
    against_labels = ["AGAINST" for i in range(number_of_tweets)]
    train_labels = train_labels + against_labels

# Lemmatizing the tweets
if (use_lemming):
    train = ptd.lemmatizing(train_tweets)
else:
    train = train_tweets

print "\t - Length of train set and labels should be the same: " + str(len(train)) + " == " + str(len(train_labels))


# ****** Creating test set (not used if cross_validation.train_test_split is used below) *******************************
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
print "\t - Length of test set and labels should be the same: " + str(len(test)) + " == " + str(len(test_labels))


# ****** Create a bag of words from the training set *******************************************************************
print "\nCreating the bag of words..."

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer1Gram = CountVectorizer(analyzer = "word",         # Split the corpus into words
                                    ngram_range = (1,1),     # N-gram: (1,1) = unigram, (2,2) = bigram
                                    stop_words = "english")  # Built-in list of english stop words.)

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer2Gram = CountVectorizer(analyzer = "word",         # Split the corpus into words
                                    ngram_range = (2,2),     # N-gram: (1,1) = unigram, (2,2) = bigram
                                    stop_words = "english")  # Built-in list of english stop words.)
# Stop words list can be found at:
#https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/stop_words.py
vectorizer3Gram = CountVectorizer(analyzer = "word",         # Split the corpus into words
                                    ngram_range = (3,3),     # N-gram: (1,1) = unigram, (2,2) = bigram
                                    stop_words = "english")  # Built-in list of english stop words.

# fit_transform() does two functions: First, it fits the model and learns the vocabulary;
# second, it transforms our training data into feature vectors. The input to fit_transform
# should be a list of strings.
if use_tf_idf:
    print "\nApplying TF*IDF...\n"
    # Transforming to a matrix with counted number of words
    unigram_train_features = vectorizer1Gram.fit_transform(train)
    unigram_test_features = vectorizer1Gram.transform(test)

    # Creating a TF*IDF transformer
    tfidf_transformer = TfidfTransformer()

    # Transforming the count matrix to the inverse (TD*IDF)
    train_data_features = tfidf_transformer.fit_transform(unigram_train_features)
    test_data_features = tfidf_transformer.transform(unigram_test_features)

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()
    test_data_features = test_data_features.toarray()

    if use_bigram:
        bigram_train_data_features = vectorizer2Gram.fit_transform(train)
        bigram_test_data_features = vectorizer2Gram.transform(test)

        bigram_train_data_features = tfidf_transformer.fit_transform(bigram_train_data_features)
        bigram_test_data_features = tfidf_transformer.transform(bigram_test_data_features)

        train_data_features = np.c_[train_data_features, bigram_train_data_features.toarray()]
        test_data_features = np.c_[test_data_features, bigram_test_data_features.toarray()]

    if use_trigram:
        trigram_train_data_features = vectorizer3Gram.fit_transform(train)
        trigram_test_data_features = vectorizer3Gram.transform(test)

        trigram_train_data_features = tfidf_transformer.fit_transform(trigram_train_data_features)
        trigram_test_data_features = tfidf_transformer.transform(trigram_test_data_features)

        train_data_features = np.c_[train_data_features, trigram_train_data_features.toarray()]
        test_data_features = np.c_[test_data_features, trigram_test_data_features.toarray()]
else:
    # Transforming to a matrix with counted number of words
    train_data_features = vectorizer1Gram.fit_transform(train)
    test_data_features = vectorizer1Gram.transform(test)

    #freqs = [(word, train_data_features.getcol(idx).sum()) for word, idx in vectorizer1Gram.vocabulary_.items()]
    #print sorted (freqs, key = lambda x: -x[1])[:10]

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()
    test_data_features = test_data_features.toarray()

    if use_bigram:
        bigram_train_data_features = vectorizer2Gram.fit_transform(train)
        bigram_test_data_features = vectorizer2Gram.transform(test)

        train_data_features = np.c_[train_data_features, bigram_train_data_features.toarray()]
        test_data_features = np.c_[test_data_features, bigram_test_data_features.toarray()]

    if use_trigram:
        trigram_train_data_features = vectorizer3Gram.fit_transform(train)
        trigram_test_data_features = vectorizer3Gram.transform(test)

        train_data_features = np.c_[train_data_features, trigram_train_data_features.toarray()]
        test_data_features = np.c_[test_data_features, trigram_test_data_features.toarray()]


# ******* Adding additional features ***********************************************************************************
if features_used > 0:
    print "Addding features. A total of: " + str(features_used)
    if use_negation:
        print "\t- Presens of negation in tweet..."
    if use_lengthOfTweet:
        print "\t- Length of tweet..."
    if use_numberOfTokens:
        print "\t- Number of tokens in tweet..."
    if use_numberOfCapitalWords:
        print "\t- Number of capital words in tweet..."
    if use_numberOfPunctMarks:
        print "\t- Number of punctuation marks in tweet..."
    if use_numberOfLengtheningWord:
        print "\t- Number of lengthened words in tweet..."
    if use_sentimentAnalyzer:
        print "\t- Using sentiment analyzer..."
    if use_posAndNegWord:
        print "\t- Number of pos and neg words"
    if use_numberOfPronouns:
        print "\t- Number of pronouns in tweet"

    trainTable = []
    testTable = []
    for i in range(len(train)):

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
                #trainTable.append([sentiments['compound']])
                trainTable.append([sentiments['neg'], sentiments['neu'], sentiments['pos']])

        if use_numberOfPronouns:
            if len(trainTable) > i:
                trainTable[i].append(ptd.getNumberOfPronouns(ptd.getPOStags(word_tokenize(train[i]))))
            else:
                trainTable.append([ptd.getNumberOfPronouns(ptd.getPOStags(word_tokenize(train[i])))])

        if use_posAndNegWord:
            if len(trainTable) > i:
                trainTable[i].append(ptd.getPosAndNegWords(train[i])[0])    # [0] = Pos
                trainTable[i].append(ptd.getPosAndNegWords(train[i])[1])    # [1] = neg
            else:
                trainTable.append([ptd.getPosAndNegWords(train[i])])


    ##### Test set features #####
    for i in range(len(test)):
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
                #testTable.append([sentiments['compound']])
                testTable.append([sentiments['neg'], sentiments['neu'], sentiments['pos']])

        if use_numberOfPronouns:
            if len(testTable) > i:
                testTable[i].append(ptd.getNumberOfPronouns(ptd.getPOStags(word_tokenize(train[i]))))
            else:
                testTable.append([ptd.getNumberOfPronouns(ptd.getPOStags(word_tokenize(train[i])))])

        if use_posAndNegWord:
            if len(testTable) > i:
                testTable[i].append(ptd.getPosAndNegWords(train[i])[0])    # [0] = Pos
                testTable[i].append(ptd.getPosAndNegWords(train[i])[1])    # [1] = neg
            else:
                testTable.append([ptd.getPosAndNegWords(train[i])])


    print trainTable[:2]
    train_data_features = np.c_[train_data_features, trainTable]
    test_data_features = np.c_[test_data_features, testTable]

# ******* Train SVM classifier using bag of words **********************************************************************
print "\nCreating and training classifiers: - Time used so far (in sec): " + str(time.time()-start_time)
if use_svm:
    print "\t - Train SVM classifier..."
    #Create a SVM model
    #clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
    #                     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
    #                     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
    #                     verbose=0)
    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                  gamma='auto', kernel='linear', max_iter=-1, probability=True,
                  random_state=None, shrinking=True, tol=0.001, verbose=False)

    # Fit model
    clf.fit(train_data_features, train_labels)

# ******* Train Multinomial Naive Bayes classifier using bag of words **********************************************************************
if use_nb:
    print "\t - Train Multinomial Naive Bayes.."
    clf_nb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    clf_nb.fit(train_data_features, train_labels)


# ******* Train dummy classifier ***************************************************************************************
if use_dummy:
    print "\t - Train dummy classifier..."
    clf_dummy = dummy.DummyClassifier(strategy='most_frequent', random_state=None, constant=None)
    clf_dummy.fit(train_data_features, train_labels)


# ******* Cross validation *********************************************************************************************
print "\nRetrieving cross validation scores..."
#all_features = np.vstack([train_data_features, test_data_features])
#all_labels = train_labels + test_labels

kf = cross_validation.StratifiedKFold(train_labels, n_folds=5, shuffle=True, random_state=1)

if use_crossval_score:
    if use_svm:
        print "Cross validation scores for SVM: "
        score = cross_validation.cross_val_score(clf, train_data_features, train_labels, cv=kf, scoring='f1_macro')
        print score
        print "Cross validation mean for SVM: "
        print score.mean()

    if use_nb:
        print "Cross validation scores for Naive Bayes: "
        score_nb = cross_validation.cross_val_score(clf_nb, train_data_features, train_labels, cv=kf, scoring='f1_macro')
        print score_nb
        print "Cross validation mean for Naive Bayes: "
        print score_nb.mean()

# ******* Predicting test labels ***************************************************************************************
print "Predicting test labels for classifiers..."
if use_svm:
    print "\n- Linear SVM"
    svm_predictions = clf.predict(test_data_features)

    if print_classification_report:
        pred_stances_svm = cross_validation.cross_val_predict(clf, train_data_features, train_labels, cv=kf)
        print classification_report(train_labels, pred_stances_svm, digits=4)

        macro_f_svm = fbeta_score(train_labels, pred_stances_svm, 1.0,
                              labels=['AGAINST', 'FAVOR'], average='macro')
        print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f_svm)

if use_dummy:
    print "\n\- Dummy"
    dummy_predictions = clf_dummy.predict(test_data_features)

    if print_classification_report:
        pred_stances_dummy = cross_validation.cross_val_predict(clf_dummy, train_data_features, train_labels, cv=kf)
        print classification_report(train_labels, pred_stances_dummy, digits=4)

        macro_f_dummy = fbeta_score(train_labels, pred_stances_dummy, 1.0,
                              labels=['AGAINST', 'FAVOR'], average='macro')
        print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f_dummy)

if use_nb:
    print "\n- Naive bayes"
    nb_predictions = clf_nb.predict(test_data_features)

    if print_classification_report:
        pred_stances_nb = cross_validation.cross_val_predict(clf_nb, train_data_features, train_labels, cv=kf)
        print classification_report(train_labels, pred_stances_nb, digits=4)

        macro_f_nb = fbeta_score(train_labels, pred_stances_nb, 1.0,
                              labels=['AGAINST', 'FAVOR'], average='macro')
        print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f_nb)

print "\nVoting classifier:"
vot_clf = VotingClassifier(estimators=[('SVM', clf),
                                       ('NB', clf_nb)],
                           voting='soft',
                           weights=[1, 1])

pred_stances_vot = cross_validation.cross_val_predict(vot_clf, train_data_features, train_labels, cv=kf)
print classification_report(train_labels, pred_stances_vot, digits=4)

macro_f_vot = fbeta_score(train_labels, pred_stances_vot, 1.0,
                      labels=['AGAINST', 'FAVOR'], average='macro')
print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f_vot)


# ******* Probabilities ************************************************************************************************
# To use the probabilities uncomment the lines 335 to 338 and then comment line 340.
minConfidence = 0.75
#if use_svm:
#    svm_predictions_probabilities = clf.predict_proba(test_data_features)
#if use_nb:
#    NB_predictions_probabilities = clf_nb.predict_proba(test_data_features)
#print svm_predictions_probabilities #against, favor, none


#************ Write to file ********************************************************************************************
if use_writeToFile:
    print "Writing gold and guesses to file..."
    data_file = test_data

    # Erwin annotated test data:
    annotated = ptd.getAnnotatedData()
    erwinsAnnotated = ptd.convertNumberStanceToText([int(row[1]) for row in annotated])

    svm_guess_file = write.initFile("guess_svm")
    dummy_guess_file = write.initFile("guess_dummy")
    nb_guess_file = write.initFile("guess_nb")
    gold_file = write.initFile("gold")

    for index in range(len(svm_predictions)):
        #if max(svm_predictions_probabilities[index]) > minConfidence:
        #   write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], svm_predictions[index], svm_guess_file)
        #else:
        #   write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], "NONE", svm_guess_file)

        write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], svm_predictions[index], svm_guess_file)
        write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], nb_predictions[index], nb_guess_file)
        write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], dummy_predictions[index], dummy_guess_file)
        #write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], data_file[index][3], gold_file)
        write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], erwinsAnnotated[index], gold_file)

    svm_guess_file.close()
    dummy_guess_file.close()
    gold_file.close()
    nb_guess_file.close()


    #*********** Evaluate the result with the given SemEval16 script *******************************************************
    print "\nResults:\n"
    print "Dummy prediction score: "
    os.system("perl eval.pl gold.txt guess_dummy.txt")
    print "SVM prediction score: "
    os.system("perl eval.pl gold.txt guess_svm.txt")
    print "Naive Bayes prediction score: "
    os.system("perl eval.pl gold.txt guess_nb.txt")

#*********** Printing total time running *******************************
print("\n--- %s seconds ---" % (time.time() - start_time))
