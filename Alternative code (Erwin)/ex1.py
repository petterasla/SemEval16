import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.metrics import fbeta_score
# import matplotlib
# import matplotlib.pyplot as plt
#
#
# matplotlib.style.use('ggplot')
# %matplotlib inline

data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)

targets = list(data.Target.unique()) + ['All']

classifiers = MultinomialNB(), LinearSVC(C=0.1)

results = pd.DataFrame(np.zeros(len(targets),
                                dtype=[('MultinomialNB', 'f'),
                                       ('LinearSVC', 'f')]),
                                index=targets)

for target in targets:
    print 80 * "="
    print target
    print 80 * "="

    target_data = data[data.Target == target] if target != 'All' else data

    cv = StratifiedKFold(target_data.Stance, n_folds=5, shuffle=True,
                         random_state=1)

    for clf in classifiers:
        print clf, '\n'
        pipeline = Pipeline([('vect', CountVectorizer(decode_error='ignore')),
                             ('clf', clf)])

        pred_stances = cross_val_predict(pipeline, target_data.Tweet,
                                         target_data.Stance, cv=cv)

        print classification_report(target_data.Stance, pred_stances, digits=4)

        macro_f = fbeta_score(target_data.Stance, pred_stances, 1.0,
                              labels=['AGAINST', 'FAVOR'], average='macro')

        print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f)

        clf_name = str(clf).split('(')[0]
        results.at[target, clf_name] = macro_f


print results
# results.plot(kind='barh')
# axes = plt.gca()
# axes.set_xlim([0,1.0])
# axes.set_xlabel('Macro F')