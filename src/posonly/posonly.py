import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***

    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    theta = np.zeros(x_train.shape[1])
    clf = LogisticRegression(step_size=1e-5, max_iter=10000000, eps=1e-5,
                             theta_0=theta, verbose=True)
    clf.fit(x_train, t_train)
    t_pred = clf.predict(x_test)
    np.savetxt(output_path_true, t_pred)
    util.plot(x_test, t_test, clf.theta, 'test1.png', correction=1)

    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    clf = LogisticRegression(step_size=1e-5, max_iter=10000000, eps=1e-5,
                             theta_0=theta, verbose=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    np.savetxt(output_path_naive, y_pred)
    util.plot(x_test, t_test, clf.theta, 'test2.png', correction=1)

    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    x_eval, y_eval = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    x_eval, t_eval = util.load_dataset(valid_path, label_col='t', add_intercept=True)

    y_pred = clf.predict(x_eval)
    alpha = np.mean(y_pred[y_eval == 1])
    y_pred = y_pred / alpha
    np.savetxt(output_path_adjusted, y_pred)
    util.plot(x_eval, t_eval, clf.theta, 'test3.png', correction=alpha)
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
