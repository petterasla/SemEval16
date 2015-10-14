from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm, cross_validation
import processTrainingData as ptd

TOPIC = "Climate Change is a Real Concern"
list_of_strings = []
list_of_labels = []

# ****** Reading in the data set ******
print "Creating list of strings..."
string_lists = ptd.processData("semeval2016-task6-trainingdata.txt")

for i in range(len(string_lists)-1):
    if string_lists[i][1] == TOPIC:
        list_of_strings.append(string_lists[i][2])
        if string_lists[i][3] == "AGAINST":
            list_of_labels.append(0)
        elif string_lists[i][3] == "NONE":
            list_of_labels.append(1)
        else:
            list_of_labels.append(2)

# ****** Creating test and training sets ******
train = list_of_strings[0:75]
test = list_of_strings[-75:]

train_labels = list_of_labels[0:75]
test_labels = list_of_labels[75:]

# ****** Create a bag of words from the training set ******
print "Creating the bag of words..."

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word", \
                             tokenizer = None, \
                             preprocessor = None, \
                             stop_words = None, \
                             max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(train)

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()

# # Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(test)
test_data_features = test_data_features.toarray()


# ******* Train SVM classifier using bag of words *******
print "Train classifier using SVM..."
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
              gamma=0.0, kernel='linear', max_iter=-1, probability=False,
              random_state=None, shrinking=True, tol=0.001, verbose=False)
# clf.fit(train_data_features, train_labels)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data_features, train_labels, test_size=0.51, random_state=0)
clf.fit(X_train, y_train)

# ******* Predicting test labels *******
print "Predicting test labels..."
#clf.predict(test_data_features, test_labels)
print clf.score(X_test, y_test)