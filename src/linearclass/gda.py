import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA(step_size=0.01, max_iter=10000, eps=1e-5, theta_0=None, verbose=False)
    clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    util.plot(x_eval, y_eval, clf.theta, 'test.png', correction=1.0)
    np.savetxt(save_path, y_pred)
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        if self.theta == None:
            self.theta = np.zeros(x.shape[1] + 1)
        # Find phi, mu_0, mu_1, and sigma
        phi = sum(y == 1) / y.size
        mu_0 = np.mean(x[y == 0], 0)
        mu_1 = np.mean(x[y == 1], 0)
        sigma = (1 / y.size) * (np.matmul((x[y == 0] - mu_0).T, (x[y == 0] - mu_0)) \
                                + np.matmul((x[y == 1] - mu_0).T, (x[y == 1] - mu_1)))

        # Write theta in terms of the parameters
        self.theta[0] = np.log(phi / (1 - phi)) + 0.5 * np.matmul(mu_0.T, np.matmul(sigma, mu_0)) - 0.5 * np.matmul(mu_1.T, np.matmul(sigma, mu_1))
        self.theta[1:] = np.linalg.solve(sigma, mu_1 - mu_0)
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        z = np.matmul(x, self.theta)
        sig_z = np.ones_like(z)/(1 + np.exp(-z))
        return sig_z
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
