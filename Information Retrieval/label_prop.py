# Import
import numpy as np
import BaselineSystem.processTrainingData as ptd
import process_tweets as twitter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.semi_supervised import label_propagation

def propagateLabels(dataset, filename):

    # LOAD LABELED DATA
    labeled_data = ptd.getTopicData(dataset)
    #labeled_data = labeled_data[0:len(labeled_data)]
    labeled_data = labeled_data[74:95]  # Use only a small portion of the data as for labelpropagation

    labeled_tweets = ptd.getAllTweets(labeled_data)
    labeled_tweets_targets = ptd.getAllStances(labeled_data)

    # LOAD UNLABELED DATA
    unlabeled_tweets = twitter.readTweets(filename)
    #unlabeled_tweets = []
    unlabeledTweets = [662022121904840704, 662020009183551488, 662016401067155456, 662020009783373824, 662016718303195136,
                       662015471261245441, 662015310367752192, 662018031292256256, 662018963916722176, 662015805635325952]
    t = twitter.readTweetsAndIDs("stream__climatechange_clean")
    for tw in unlabeledTweets:
        for id in t:
            if tw == id[0]:
                unlabeled_tweets.append(id[1])

    # MERGE UNLABELED AND LABELED DATA
    all_tweets = labeled_tweets + unlabeled_tweets
    all_labels = ptd.convertStancesToNumbers(labeled_tweets_targets) + [-1 for i in range(len(unlabeled_tweets))]

    # CREATE FEATURES
    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",         # Split the corpus into words
                                 ngram_range = (1,1),       # N-gram: (1,1) = unigram, (2,2) = bigram
                                 stop_words = "english")    # Built-in list of english stop words.)

    all_tweets_features = vectorizer.fit_transform(all_tweets)

    # CREATE MODEL FOR LABEL PROPAGATION
    labelProp = label_propagation.LabelSpreading()  # Can also use .LabelPropagation()
    model = labelProp.fit(all_tweets_features.toarray(), all_labels)

    # PRINT SOME STATISTICS AND DEBUG INFO
    #sum = 0
    #table = (all_labels==model.transduction_)
    #for i in table:
    #    if i == True:
    #        sum += 1

    #print sum
    #print len(labeled_data)
    print(model.transduction_[:len(labeled_data)])
    print(model.transduction_[len(labeled_data):])
    #print len(all_tweets_features[0].toarray()[0])


    # TEST
    # test_tweets = []
    # tweets = [662022121904840704, 662020009183551488, 662016401067155456, 662020009783373824, 662016718303195136, 662015471261245441, 662015310367752192, 662018031292256256, 662018963916722176, 662015805635325952]
    # t = twitter.readTweetsAndIDs("stream__climatechange_clean")
    # for tw in tweets:
    #     for id in t:
    #         if tw == id[0]:
    #             test_tweets.append(id[1])
    #
    # print model.predict(vectorizer.fit_transform(test_tweets))

    # STORE PREDICTIONS
    predicted_labels = model.transduction_[len(labeled_data):]

    # CREATE TABLE OF PREDICTIONS AND TWEET
    predictions = [-1 for i in range(len(predicted_labels))]
    for i in range(len(predicted_labels)):
        predictions[i] = [predicted_labels[i], unlabeled_tweets[i]]

    return predictions


# TESTING
results = propagateLabels("Climate Change is a Real Concern", "stream__climate_clean")
for result in results:
    print result