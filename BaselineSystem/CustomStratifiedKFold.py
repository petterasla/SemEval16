import warnings
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.fixes import bincount
from sklearn.cross_validation import StratifiedKFold, KFold

class CustomStratifiedKFold(StratifiedKFold):
    def __init__(self, y, n_folds=3, shuffle=False, random_state=None, samples=None):
        super(CustomStratifiedKFold, self).__init__(y, n_folds, shuffle, random_state)

        if samples is None:
            y_include = y
            y_not_include = None
        else:
            y_include = y[0:samples]
            y_not_include = y[samples:]

        y_include = np.asarray(y_include)
        n_samples = y_include.shape[0]
        unique_labels, y_inversed = np.unique(y_include, return_inverse=True)
        label_counts = bincount(y_inversed)
        min_labels = np.min(label_counts)

        if self.n_folds > min_labels:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is too few. The minimum"
                           " number of labels for any class cannot"
                           " be less than n_folds=%d."
                           % (min_labels, self.n_folds)), Warning)

        # don't want to use the same seed in each label's shuffle
        if self.shuffle:
            rng = check_random_state(self.random_state)
        else:
            rng = self.random_state

        # pre-assign each sample to a test fold index using individual KFold
        # splitting strategies for each label so as to respect the
        # balance of labels
        per_label_cvs = [
            KFold(max(c, self.n_folds), self.n_folds, shuffle=self.shuffle, random_state=rng) for c in label_counts]
        test_folds = np.zeros(n_samples, dtype=np.int)

        for test_fold_idx, per_label_splits in enumerate(zip(*per_label_cvs)):
            for label, (_, test_split) in zip(unique_labels, per_label_splits):
                label_test_folds = test_folds[y_include == label]
                # the test split can be too big because we used
                # KFold(max(c, self.n_folds), self.n_folds) instead of
                # KFold(c, self.n_folds) to make it possible to not crash even
                # if the data is not 100% stratifiable for all the labels
                # (we use a warning instead of raising an exception)
                # If this is the case, let's trim it:
                test_split = test_split[test_split < len(label_test_folds)]
                label_test_folds[test_split] = test_fold_idx
                test_folds[y_include == label] = label_test_folds

        self.test_folds = test_folds
        self.y = y_include

    def _iter_test_masks(self):
        for i in range(self.n_folds):
            yield self.test_folds == i

    def __repr__(self):
        return '%s.%s(labels=%s, n_folds=%i, shuffle=%s, random_state=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.y,
            self.n_folds,
            self.shuffle,
            self.random_state,
        )

    def __len__(self):
        return self.n_folds

