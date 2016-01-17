import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn.metrics import fbeta_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.feature_extraction.text import CountVectorizer
from custom_cross_validation import cross_val_predict   # custom scikit cross validation
import cross_val_generator                              # Custom fold generation

# Set up
topic = "Climate Change is a Real Concern"

original_data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
original_targets = list(original_data.Target.unique()) + ['All']

label_prop_data = pd.read_csv(open('label_propagated_data.txt'), '\t', index_col=0)
label_prop_targets = list(original_data.Target.unique()) + ['All']

include = len(original_data[original_data.Target == topic])
target_data = pd.concat([original_data[original_data.Target == topic], label_prop_data[label_prop_data.Target == topic]], axis=0)

data = original_data[original_data.Target == topic]

# Label prop KFold
kf = cross_val_generator.generateFolds(target_data.Stance, n_folds=5, shuffle=False,
                                       random_state=1, exclude_from_test=include)
# KFold without label prop
kf1 = StratifiedKFold(data.Stance, n_folds=5, shuffle=True, random_state=1)

# SVM
clf_svm = svm.LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,
                        intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                        multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                        verbose=0)

# Pipeline
pipeline_svm = make_pipeline(
    make_union(
        CountVectorizer(decode_error='ignore')
    ),
    clf_svm)

# Predictions for label prop
pred_stances_svm = cross_val_predict(pipeline_svm, target_data.Tweet, target_data.Stance, cv=kf)

print classification_report(data.Stance, pred_stances_svm, digits=4) # Use only labeled data in prediction report

macro_f_svm = fbeta_score(data.Stance, pred_stances_svm, 1.0, # Use only labeled data
                          labels=['AGAINST', 'FAVOR'],
                          average='macro')
print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f_svm)


# Predictions without label prop
pred_stances_svm_org = cross_val_predict(pipeline_svm, data.Tweet, data.Stance, cv=kf1)

print classification_report(data.Stance, pred_stances_svm_org, digits=4)

macro_f_svm = fbeta_score(data.Stance, pred_stances_svm_org, 1.0,
                          labels=['AGAINST', 'FAVOR'],
                          average='macro')
print 'macro-average of F-score(FAVOR) and F-score(AGAINST): {:.4f}\n'.format(macro_f_svm)
