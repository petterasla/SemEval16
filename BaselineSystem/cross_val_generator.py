import numpy as np
from sklearn.cross_validation import StratifiedKFold

def generateFolds(labels, n_folds=3, exclude_from_test=None, shuffle=False, random_state=None):

    labels_include = labels[:exclude_from_test]
    labels_not_include = labels[exclude_from_test:]

    cv_train_test = StratifiedKFold(labels_include, n_folds, shuffle, random_state)
    cv_train = StratifiedKFold(labels_not_include, n_folds, shuffle, random_state)

    train_indeces = []
    test_indeces = []

    iterator = []
    for train_index, test_index in cv_train_test:
        iterator.append((train_index, test_index))
        test_indeces.append(test_index)
        #train_indeces.append(np.concatenate((train_index, [i for i in range(len(labels_include), len(labels_include)+len(labels_not_include))])))

    counter = 0
    for train_index, test_index in cv_train:
        train_indeces.append(np.concatenate((iterator[counter][0], train_index + len(labels_include)), axis=0))
        #final_iterator.append((total_train, iterator[counter][1]))
        counter += 1

    final_iterator = zip(train_indeces, test_indeces)
    print len(labels_include)
    print len(labels_not_include)
    for train_index, test_index in final_iterator:
        print 80*"="
        print "Length train:    ",len(train_index)
        print "Length test:     ", len(test_index)
        print "Lenght total:    ", len(train_index)+len(test_index)
        print "Max test index:  ", max(test_index)
        print "Max train index: ", max(train_index)
        
    return final_iterator