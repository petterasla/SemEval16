from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, cross_validation
import processTrainingData as ptd

TOPIC = "All"
TOPIC1 = "Atheism"
TOPIC2 = "Climate Change is a Real Concern"
TOPIC3 = "Feminist Movement"
TOPIC4 = "Hillary Clinton"
TOPIC5 = "Legalization of Abortion"

data = TOPIC2

# ****** Creating test and training sets ******
print "Creating test and training sets with topic: " + str(data)

tweets = ptd.getAllTweets(ptd.getTopicData(data))
train = tweets
train_labels = ptd.getAllStances(ptd.getTopicData(data))

print "len of train set and labels: " + str(len(train)) + " == " + str(len(train_labels))


# ****** Create a bag of words from the training set ******
print "Creating the bag of words..."

# Initialize the "TfidfVectorizer" object, which is scikit-learn's bag of words tool.
# Convert a collection of raw documents to a matrix of TF-IDF features.
# Equivalent to CountVectorizer followed by TfidfTransformer.
tfidf_vectorizer = TfidfVectorizer(input=train,
                                   analyzer = "word",
                                   preprocessor = None,
                                   max_features = 5000)

train_data_features = tfidf_vectorizer

# Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()

# # Get a bag of words for the test set, and convert to a numpy array
#test_data_features = vectorizer.transform(test)
#test_data_features = test_data_features.toarray()


# ******* Train SVM classifier using bag of words *******
print "Train classifier using SVM..."
# Create a SVM model
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
              gamma=0.0, kernel='linear', max_iter=-1, probability=False,
              random_state=None, shrinking=True, tol=0.001, verbose=False)

# Divide into test and train partition
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data_features, train_labels, test_size=0.25, random_state=0)

# Fit model
clf.fit(X_train, y_train)


# ******* Predicting test labels *******
print "Predicting test labels..."
print "Size of test set: " + str(len(X_test))
print "Size of train set: " +str(len(X_train))

# Calcualte score using k-fold cross validation
scores = cross_validation.cross_val_score(clf, train_data_features, train_labels, cv=5, n_jobs=-1)

print 'Score from CV: ' + str(scores)
print 'Score from test set: ' + str(clf.score(X_test, y_test))