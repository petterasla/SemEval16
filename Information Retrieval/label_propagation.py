# Import
import numpy as np
import BaselineSystem.processTrainingData as ptd
import process_tweets as twitter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.semi_supervised import label_propagation

def propagateLabels(dataset, filename):
    #dataset = "Climate Change is a Real Concern"
    #filename = "stream__climate_clean"

    # Load test and training data
    labeled_data = ptd.getTopicData(dataset)
    labeled_data = labeled_data[0:len(labeled_data)/4]

    labeled_tweets = ptd.getAllTweets(labeled_data)
    labeled_tweets_targets = ptd.getAllStances(labeled_data)

    unlabeled_tweets = twitter.readTweets(filename)

    all_tweets = labeled_tweets + unlabeled_tweets
    all_labels = ptd.convertStancesToNumbers(labeled_tweets_targets) + [-1 for i in range(len(unlabeled_tweets))]

    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",         # Split the corpus into words
                                 ngram_range = (1,1),       # N-gram: (1,1) = unigram, (2,2) = bigram
                                 stop_words = "english")    # Built-in list of english stop words.)
    all_tweets_features = vectorizer.fit_transform(all_tweets)

    # Set number of desired labeled points
    n_total_samples = len(all_tweets)
    n_labeled_points = len(labeled_tweets)

    # Remove labels
    unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]

    # Propagate labels
    lp_model = label_propagation.LabelSpreading(kernel='rbf', gamma=0.17, n_neighbors=7, alpha=0.2, max_iter=50, tol=0.01)
    #lp_model = label_propagation.LabelSpreading(gamma=0.25, max_iter=5)

    lp_model.fit(all_tweets_features.toarray(), all_labels)

    predicted_labels = lp_model.transduction_[unlabeled_indices]

    sum = 0
    table = (all_labels==lp_model.transduction_)
    for i in table:
        if i == True:
            sum += 1

    print sum
    print len(labeled_data)

    print lp_model.transduction_[:len(labeled_data)]
    print lp_model.transduction_[len(labeled_data):]

    predictions = [-1 for i in range(len(predicted_labels))]
    for i in range(len(predicted_labels)):
        predictions[i] = [predicted_labels[i], unlabeled_tweets[i]]

    return predictions
    #return lp_model.transduction_, all_tweets



## USE THIS FOR TESTING
results = propagateLabels("Climate Change is a Real Concern", "stream__climate_clean")
#for result in results:
#    print result