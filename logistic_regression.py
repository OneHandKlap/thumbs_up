import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.20, random_state = 0)
from sklearn.preprocessing import StandardScaler  #is this needed since x value is the words, y is positive or negative
sc_x = StandardScaler()
x_train = sc.x_fit_transform(x_train)
x_test = sc.x_transform(x_test)


class LogisticRegression(object):

    dataset = pd.read_csv('train_small.csv')
    dataset.head()

    #input
    x = dataset.iloc[:, 0].values

    #output
    y = dataset.iloc[:, 2].values 

    #training the Logistic Regression Model
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression (random_state = 0)
    classifier.fit(x_train, y_train)

    y_predictions = classifier.predict(x_test)

    #output Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix (y_predictions, y_test)
    print("Confusion Matrix : \n", cm)

    #
    from sklearn.metrics import accuracy_score
    # accuracy_score (y_predictions, y_test)
    print ("Accuracy : ", accuracy_score(y_predictions, y_test))

    #
    from matplotlib.colors import ListedColourmap
    x_set, y_set = x_test, y_test
    X2, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max + 1, step = 0.01))

    plt.contourf(X1, X2, classifier.predict (
        np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColourmap(('black', 'white')))

    plt.x_lim(X1.min(), X1.max())
    plt.y_lim(X2.min(), X2.max())

    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColourmap (('black', 'white')) (i), label = j)

    plt.scatter(x, y)
    plt.title('Classifier (Test set)')
    plt.x_label('IMBD')
    plt.y_label('Review')
    plt.legend()
    plt.plot(X, y_predictions, color = 'red')
    plt.show()


