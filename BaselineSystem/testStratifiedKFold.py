import pandas as pd
from CustomStratifiedKFold import CustomStratifiedKFold
from sklearn.cross_validation import StratifiedKFold

original_data = pd.read_csv(open('semeval2016-task6-trainingdata.txt'), '\t', index_col=0)
original_targets = list(original_data.Target.unique()) + ['All']
#original_data.to_csv('out.csv')
#print original_data

label_prop_data = pd.read_csv(open('label_propagated_data.txt'), '\t', index_col=0)
label_prop_targets = list(original_data.Target.unique()) + ['All']
#label_prop_data.to_csv('out.csv')
#print label_prop_data

all_data = pd.concat([original_data, label_prop_data], axis=0)
#all_data.to_csv('out.csv')
#print all_data

include = [1 for i in range(len(original_data))]
not_include = [-1 for j in range(len(label_prop_data))]
testing_data = include + not_include

target_data = original_data[original_data.Target == "Climate Change is a Real Concern"]

kf1 = StratifiedKFold(all_data.Stance, n_folds=5, shuffle=False, random_state=1)
kf2 = CustomStratifiedKFold(all_data.Stance, n_folds=5, shuffle=False, random_state=1, samples=len(original_data))
# print len(kf1)
print "KF1"
for train_index, test_index in kf1:
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
# print len(kf2)
print "KF2"
for train_index, test_index in kf2:
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
