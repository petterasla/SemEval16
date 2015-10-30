from sklearn.feature_extraction.text import CountVectorizer
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

t = TOPIC           # Select a topic that will be used for training and testing
ratio = 0.2         # test ratio from data set


# ****** Creating training and test set ******
print "Creating test and training sets with topic: " + str(t)
# Splitting data into train and test data.
train_data, test_data = ptd.train_test_split(ptd.getTopicData(t), ratio)

# Getting all the tweets and removing hashtags and @ tags.
train_tweets = ptd.getAllTweetsWithoutHashOrAlphaTag(ptd.getAllTweets(train_data))
# Getting all the hashtags from each tweet
train_hashtags = ptd.getAllHashtags(ptd.getAllTweets(train_data))
# Parsing the hashtags to words, then add them to the tweet again
for index in range(len(train_hashtags)):
    words = ptd.decryptHashtags(train_hashtags[index])
    for word in words:
        train_tweets[index] = train_tweets[index] + " " + word + " "

# Training and test set
train = train_tweets
train_labels = ptd.getAllStances(train_data)
print "Length of train set and labels should be the same: " + str(len(train)) + " == " + str(len(train_labels))

# ************ Creating test set (not used if cross_validation.train_test_split is used below)************
test_tweets = ptd.getAllTweetsWithoutHashOrAlphaTag(ptd.getAllTweets(test_data))
test_hashtags = ptd.getAllHashtags(ptd.getAllTweets(test_data))
for index in range(len(test_hashtags)):
    words = ptd.decryptHashtags(test_hashtags[index])
    for word in words:
        test_tweets[index] = test_tweets[index] + " " + word + " "

test = test_tweets
test_labels = ptd.getAllStances(test_data)
print "Length of test set and labels should be the same: " + str(len(test)) + " == " + str(len(test_labels))
# ****** Create a bag of words from the training set ******
print "Creating the bag of words..."

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",         # Split the corpus into words
                             ngram_range = (1,1),       # N-gram: (1,1) = unigram, (2,2) = bigram
                             stop_words = "english",    # Built-in list of english stop words.
                             max_features = 5000)

# fit_transform() does two functions: First, it fits the model and learns the vocabulary;
# second, it transforms our training data into feature vectors. The input to fit_transform
# should be a list of strings.
train_data_features = vectorizer.fit_transform(train)

# Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()


# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(test)
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
clf_dummy = dummy.DummyClassifier(strategy='stratified', random_state=None, constant=None)
clf_dummy.fit(train_data_features, train_labels)


# ******* Predicting test labels *******
print "Predicting test labels..."
svm_predictions = clf.predict(test_data_features)
dummy_predictions = clf_dummy.predict(test_data_features)

print "len of svm: " + str(len(svm_predictions))
print "len of svm: " + str(len(dummy_predictions))

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
gold_file.close()

#*********** Evaluate the result with the given SemEval16 script **************
print "\nResults:\n"
print "Dummy prediction score: "
os.system("perl eval.pl gold.txt guess_dummy.txt")
print "SVM prediction score: "
os.system("perl eval.pl gold.txt guess_svm.txt")
