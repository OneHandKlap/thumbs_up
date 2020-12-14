import pandas as pd
import numpy as np

class LogisticModel(object):

    def __init__(self,vocabulary,alpha=0.000035,eps=1e-5, max_iter=10000,thetas=None,verbose=False, threshold=0.5):
        self.vocab=vocabulary
        if thetas==None:
            self.theta=np.array([1]*len(vocabulary))

        self.alpha=alpha
        self.max_iterations=max_iter
        self.verbose=verbose
        self.threshold=threshold
        self.eps=eps

    def fit(self,train_df):
        x=train_df.iloc[:,1:-1]
        y=train_df['y']
        count=0
        while count < self.max_iterations:
            prev_theta=np.copy(self.theta)
            gradient=self._gradient(x,y)
            
            self.theta=np.add(self.theta,self.alpha*(gradient))

            loss = self._loss(x, y)
            if self.verbose:
                print('[iter: {:02d}, loss: {:.7f}]'.format(count, loss))

            if np.linalg.norm(prev_theta - self.theta) < self.eps:
                break
            count+=1


    def predict(self, test_df):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        x=test_df.iloc[:,1:-1]
        
        y_hat = self._sigmoid(x.dot(self.theta))
        return y_hat

    def _gradient(self, x, y):
        """Get gradient of J.

        Returns:
            grad: The gradient of J with respect to theta. Same shape as theta.
        """
        m, _ = x.shape

        probs = self._sigmoid(x.dot(self.theta))

        grad =x.T.dot(y-probs)

        return grad

    def _loss(self, x, y):
        """Get the empirical loss for logistic regression."""
        hx = self._sigmoid(x.dot(self.theta))
        loss = -np.mean(y * np.log(hx + self.alpha) + (1 - y) * np.log(1 - hx + self.alpha))

        return loss

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


