import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    #Step 1: Train a logistic regression classifier
    theta = np.random.randn(x_train.shape[1]) * 0.01
    clf = LogisticRegression(step_size = 0.01, max_iter=1000000, eps=1e-5,
                theta_0=theta, verbose=True)
    clf.fit(x_train, y_train)
    #Step 2: Test the model on valid_path
    x_test, y_test = util.load_dataset(valid_path, add_intercept=True)
    y_hat = clf.predict(x_test)
    #Step 3: Use np.savetxt to save predictions on test set to save_path
    np.savetxt(save_path, y_hat)
    #Step 4: Plot decision boundary on top of validation set
    util.plot(x_test, y_test, clf.theta, 'plot.png', correction=1.0)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """


        # *** START CODE HERE **
        eps = 10
        iters = 0
        del_theta = np.zeros_like(self.theta)
        while eps > self.eps and iters < self.max_iter:
            D = np.diag(self.predict(x) * (1 - self.predict(x)))
            H = np.matmul(x.T, np.matmul(D, x))
            del_theta = np.linalg.solve(H, np.dot(self.predict(x) - y, x))
            self.theta -= del_theta
            eps = np.sum(np.abs(del_theta))
            iters += 1
        self.eps = eps
        self.max_iter = iters



        # eps = 10
        # iters = 0
        # m = y.shape[0]
        # y.shape = (m, 1)
        # #print("y shape = ", y.shape)
        # update_theta = np.random.randn(x.shape[1], 1) * 0.01
        # while eps > self.eps and iters < self.max_iter:
        #     #print(self.predict(x).shape, y.shape)
        #     d_theta = np.multiply(self.predict(x) - y, x)
        #     #print("d_theta shape", d_theta.shape)
        #     temp = (self.predict(x) * (1 - self.predict(x)))
        #     #print(temp.shape)
        #     H = np.matmul(x, np.matmul(temp, x))
        #     update_theta = np.linalg.solve(H, d_theta)
        #     self.theta -= update_theta
        #     eps = np.sum(np.abs(update_theta))
        #     iters += 1
        # self.eps = eps
        # self.max_iter = iters
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        z = np.dot(x, self.theta)
        sig_z = np.ones_like(z) / (1 + np.exp(-z))
        return sig_z

        #print(self.theta.shape, x.shape)
        # z = np.dot(x, self.theta)
        # sig_z = 1 / (1 + np.exp(-z))
        # #print(sig_z.shape)
        # return sig_z
        # if sig_z > 0.5:
        #     return 1
        # if sig_z < 0.5:
        #     return 0
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
