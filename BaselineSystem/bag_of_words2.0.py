from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import svm, cross_validation
from sklearn import dummy
import processTrainingData as ptd
import writePredictionsToFile as write
import os

TOPIC = "All"
TOPIC1 = "Atheism"
TOPIC2 = "Climate Change is a Real Concern"
TOPIC3 = "Feminist Movement"
TOPIC4 = "Hillary Clinton"
TOPIC5 = "Legalization of Abortion"

topic = TOPIC2              # Select a topic that will be used as data
test_topic = TOPIC2         # Select a topic that will be used for testing
use_tf_idf = 0              # 1 = true, 0 = false
use_lemming = 0             # 1 = true, 0 = false
use_abstracts = 1           # 1 = true, 0 = false

# ****** Creating training and test set ******
print "Creating training set with topic: " + str(topic)
print "Creating test set with topic: " + str(test_topic)
# Splitting data into train and test data.
train_data, test_data = ptd.train_test_split_on_stance(ptd.getTopicData(topic), ptd.getTopicData(test_topic), 0.3, 0.3, 0.3)
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
vectorizer = CountVectorizer(analyzer = "word",         # Split the corpus into words
                             ngram_range = (1,1),       # N-gram: (1,1) = unigram, (2,2) = bigram
                             stop_words = "english")    # Built-in list of english stop words.)

# fit_transform() does two functions: First, it fits the model and learns the vocabulary;
# second, it transforms our training data into feature vectors. The input to fit_transform
# should be a list of strings.
if (use_tf_idf):
    print "Applying TF*IDF..."
    # Transforming to a matrix with counted number of words
    count = vectorizer.fit_transform(train)
    count2 = vectorizer.transform(test)

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
    train_data_features = vectorizer.fit_transform(train)
    test_data_features = vectorizer.transform(test)

    freqs = [(word, train_data_features.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
    print sorted (freqs, key = lambda x: -x[1])[:10]

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()
    test_data_features = test_data_features.toarray()


# ******* Train SVM classifier using bag of words *******
print "Train classifier using SVM..."
# Create a SVM model
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
              gamma=0.0, kernel='linear', max_iter=-1, probability=False,
              random_state=None, shrinking=True, tol=0.001, verbose=False)

# Divide into test and train partition
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data_features, train_labels, test_size=0.25, random_state=0)

# Fit model
clf.fit(train_data_features, train_labels)


# ******* Train dummy classifier ******
print "Train dummy classifier..."
clf_dummy = dummy.DummyClassifier(strategy='most_frequent', random_state=None, constant=None)
clf_dummy.fit(train_data_features, train_labels)


# ******* Predicting test labels *******
print "Predicting test labels..."
svm_predictions = clf.predict(test_data_features)
dummy_predictions = clf_dummy.predict(test_data_features)

# Calcualte score using k-fold cross validation
# scores = cross_validation.cross_val_score(clf, train_data_features, train_labels, cv=5, n_jobs=-1)

#print 'Score from CV: ' + str(scores)
#print 'Score from test set: ' + str(clf.score(test_data_features, test_labels))



#************ Write to file ******************
print "Writing gold and guesses to file..."
data_file = test_data
svm_guess_file = write.initFile("guess_svm")
dummy_guess_file = write.initFile("guess_dummy")
gold_file = write.initFile("gold")
for index in range(len(svm_predictions)):
    write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], svm_predictions[index], svm_guess_file)
    write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], dummy_predictions[index], dummy_guess_file)
    write.writePrdictionToFile(data_file[index][0], data_file[index][1], data_file[index][2], data_file[index][3], gold_file)

svm_guess_file.close()
dummy_guess_file.close()
gold_file.close()

#*********** Evaluate the result with the given SemEval16 script **************
print "\nResults:\n"
print "Dummy prediction score: "
os.system("perl eval.pl gold.txt guess_dummy.txt")
print "SVM prediction score: "
os.system("perl eval.pl gold.txt guess_svm.txt")
