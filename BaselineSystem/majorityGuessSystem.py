from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn import dummy
import processTrainingData as ptd
import writePredictionsToFile as wptf

TOPIC1 = "Climate Change is a Real Concern"
TOPIC2 = "Atheism"
TOPIC3 = "Feminist Movement"
TOPIC4 = "Hillary Clinton"
TOPIC5 = "Legalization of Abortion"

list_of_strings = []
list_of_labels = []

# ****** Reading in the data set ******
print "Creating list of strings..."
string_lists = ptd.processData("semeval2016-task6-trainingdata.txt")

for i in range(len(string_lists)-1):
    if string_lists[i][1] == TOPIC1 or \
                    string_lists[i][1] == TOPIC2 or \
                    string_lists[i][1] == TOPIC3 or \
                    string_lists[i][1] == TOPIC4 or \
                    string_lists[i][1] == TOPIC5:
        list_of_strings.append(string_lists[i][2])

        if string_lists[i][3] == "AGAINST":
            list_of_labels.append(0)
        elif string_lists[i][3] == "NONE":
            list_of_labels.append(1)
        else:
            list_of_labels.append(2)

# ****** Creating test and training sets ******
train = list_of_strings
train_labels = list_of_labels
print("Number of tweets: " + str(len(list_of_strings)))

# ****** Create a bag of words from the training set ******
print "Creating the bag of words..."

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = "word", \
                             tokenizer = None, \
                             preprocessor = None, \
                             stop_words = None, \
                             max_features = 5000)

# fit_transform() does two functions: First, it fits the model and learns the vocabulary;
# second, it transforms our training data into feature vectors. The input to fit_transform
# should be a list of strings.
train_data_features = vectorizer.fit_transform(train)

# Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()

# # Get a bag of words for the test set, and convert to a numpy array
#test_data_features = vectorizer.transform(test)
#test_data_features = test_data_features.toarray()


# ******* Train dummy classifier using bag of words *******
print "Train classifier using Majority Guess (dummy - most frequent)..."
# Create a dummy model
clf = dummy.DummyClassifier(strategy='stratified', random_state=None, constant=None)

# Divide into test and train partition
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data_features, train_labels, test_size=0.50, random_state=0)

# Fit model
clf.fit(X_train, y_train)

# ******* Predicting test labels *******
print "Predicting test labels..."
# Calcualte score using k-fold cross validation
scores = cross_validation.cross_val_score(clf, train_data_features, train_labels, cv=5)

print 'Score from CV: ' + str(scores)
print 'Score from test set: ' + str(clf.score(X_test, y_test))

"""
file = wptf.initFile("predictions")
ID = 1
for i in range(len(X_test)):
    prediction = clf.predict(X_test[i],y_test[i])
    if prediction == 0:
        prediction = "AGAINST"
    elif prediction == 1:
        prediction = "NONE"
    else:
        prediction = "FAVOUR"

    #wptf.writePrdictionToFile(ID, )
    ID += 1
file.close()
"""