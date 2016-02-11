import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

#******* Processing data *********
original_data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
atheims = original_data[original_data.Target == "Atheism"]
t =[len(atheims[atheims.Stance == "AGAINST"].Stance), len(atheims[atheims.Stance == "FAVOR"].Stance), len(atheims[atheims.Stance == "NONE"].Stance)]
targets = list(original_data.Target.unique()) + ["All"]
classifiers = [SVC(C=130),
               LinearSVC(C=0.03),
               MultinomialNB(),
               SGDClassifier(alpha=0.023),
               LogisticRegression(C=0.38, class_weight="balanced")
               ]
clf_names = [str(clf).split("(")[0] for clf in classifiers]
dtype = [(name, "f") for name in clf_names]

f_scores = pd.DataFrame(np.zeros(len(targets), dtype=dtype), index=targets)
std_errors = pd.DataFrame(np.zeros(len(targets), dtype=dtype), index=targets)
macro_f_scorer = make_scorer(fbeta_score,
                             beta=1.0,
                             labels=['AGAINST', 'FAVOR'],
                             average='macro')

for i, clf in enumerate(classifiers):
    for target in targets:
        if target != "All":
            data = original_data[original_data.Target == target]

        cv = StratifiedKFold(data.Stance, n_folds=5, shuffle=True, random_state=1)

        pipeline = Pipeline([("vectorizer", CountVectorizer(decode_error="ignore")),
                             ("classifier", clf)])
        predictions = cross_val_score(pipeline, data.Tweet, data.Stance, cv=cv, scoring=macro_f_scorer)

        print 80*"="
        print clf_names[i]
        print target
        print 80*"="

        print 'macro-average of F-score(FAVOR) and F-score(AGAINST): %0.4f (+/- %0.4f)' % (predictions.mean(), predictions.std()*2)

        f_scores.at[target, clf_names[i]] = predictions.mean()
        std_errors.at[target, clf_names[i]] = predictions.std() * 2

        print "\n\n"


# Creating a figure with two plots on one row
fig, ax1 = plt.subplots(1,1, sharex=False, sharey=False)
matplotlib.style.use('ggplot')
ax1.set_title("Baseline using dummy classifiers")

# X axis:
ax1.set_xlabel("Targets")

# Y axis
ax1.set_ylabel("Macro F-score")
ax1.set_yticks(np.arange(0, 2.0, 0.1))
ax1.set_ylim((0, 1.0))

# Other shizzle

f_scores.plot(yerr=std_errors, ax=ax1, kind="bar")
# Print that shit
plt.show()
