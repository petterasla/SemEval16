from glob import glob

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import Normalizer

from glove_transformer import GloveVectorizer

data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
data = data[data.Target == 'Climate Change is a Real Concern']
true_stances = data.Stance

cv = StratifiedKFold(true_stances, n_folds=5, shuffle=True, random_state=1)

glove_fnames = glob('*.pkl')
glove_ids = [fname.split('/')[-1].split('_')[0] for fname in glove_fnames]

for fname, glove_id in zip(glove_fnames, glove_ids):
    print 80 * '='
    print 'GLOVE VECTORS:', glove_id
    print 80 * '='

    glove_fname = 'glove/semeval2016-task6-trainingdata_climate_glove.twitter.27B.{}d.pkl'
    glove_vecs = pd.read_pickle(fname)

    glove_clf = Pipeline([('vect', GloveVectorizer(glove_vecs)),
                          ('clf', LogisticRegression(C=0.1,
                                                     solver='lbfgs',
                                                     multi_class='multinomial',
                                                     class_weight='balanced',
                                                     ))])

    char_clf = Pipeline([('vect', CountVectorizer(decode_error='ignore',
                                                  lowercase=False,
                                                  min_df=5,
                                                  ngram_range=(3, 3),
                                                  analyzer='char')),
                         ('clf', MultinomialNB())])

    word_clf = Pipeline([('vect', CountVectorizer(decode_error='ignore',
                                                  lowercase=False,
                                                  ngram_range=(2, 2))),
                         ('clf', MultinomialNB())])

    vot_clf = VotingClassifier(estimators=[('char', char_clf),
                                           ('word', word_clf),
                                           ('glove', glove_clf)],
                               voting='hard',
                               weights=[1, 1, 2])

    pred_stances = cross_val_predict(vot_clf, data.Tweet, true_stances, cv=cv)
    print classification_report(true_stances, pred_stances, digits=4)

    macro_f = fbeta_score(true_stances, pred_stances, 1.0,
                          labels=['AGAINST', 'FAVOR'], average='macro')
    print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f)