import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn import svm

# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.style.use('ggplot')
# %matplotlib inline

data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
target_data = data[data.Target == 'Climate Change is a Real Concern']
cv = StratifiedKFold(target_data.Stance, n_folds=5, shuffle=True, random_state=1)
results = pd.DataFrame(np.zeros(10000,
                                dtype=[('analyzer', 'S8'),
                                       ('ngram_range', 'S8'),
                                       ('lowercase', 'b'),
                                       ('binary', 'b'),
                                       ('min_df', 'i'),
                                       ('macro_f', 'i')]))
i = 0

for analyzer in 'word', 'char', 'char_wb':
    if analyzer == 'word':
        ngram_ranges = [(1,1), (2,2), (1,2)]
        min_dfs = 1,2
    else:
        ngram_ranges = [(2,2),(3,3),(2,3),(2,4),(2,5)]
        min_dfs = 1,2,3,5,10
    for ngram_range in ngram_ranges:
        for lowercase in True, False:
            for binary in True, False:
                for min_df in min_dfs:
                    pipeline = Pipeline([('vect', CountVectorizer(decode_error='ignore',
                                                                  binary=binary,
                                                                  lowercase=False,
                                                                  min_df=min_df,
                                                                  ngram_range=ngram_range,
                                                                  analyzer=analyzer)),
                                         ('clf', LogisticRegression(C=0.1,
                                                                    solver='lbfgs',
                                                                    multi_class='multinomial',
                                                                    class_weight='balanced',
                                                                    ))])
                    print pipeline

                    pred_stances = cross_val_predict(pipeline, target_data.Tweet, target_data.Stance, cv=cv)
                    print classification_report(target_data.Stance, pred_stances, digits=4)

                    macro_f = fbeta_score(target_data.Stance, pred_stances, 1.0,
                                          labels=['AGAINST', 'FAVOR'], average='macro')
                    print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'. \
                        format(macro_f)
                    print (analyzer, ngram_range, lowercase, binary, min_df, macro_f)
                    results.iloc[i] = (analyzer, str(ngram_range), lowercase, binary, min_df, macro_f)
                    i += 1


results = results[results.analyzer != '']
pd.set_option('display.max_rows', len(results))
results.sort_values(by='macro_f', ascending=False, inplace=True)
print results
print results[results.analyzer == 'word']