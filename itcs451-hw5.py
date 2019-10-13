"""A module to for ITCS451 assignment 5."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def load_data():
    # The following example is a dataset about
    # quiz, hw, reading hours and grade.
    return np.array([
        [1, 9, 0, 4],  # 4 is A
        [1, 9, 8, 4],  # 3 is B
        [1, 9, 9, 2],  # 2 is C
        [5, 11, 9, 1],  # 1 is D
        [2, 1, 1, 0],  # 0 is F
        [5, 11, 8, 1],
        [3, 5, 4, 1],
        [3, 6, 3, 1],
    ])
    # """
    # Return dataset for classification in a form of numpy array.
    #
    # The dataset should be in a 2D array where each column is a feature
    # and each row is a data point. The last column should be the label.
    #
    # Your data should contain only number.
    # If you have a categorical feature,  you can replace your feature with
    # "one-hot encoding". For example, a feature 'grade' of values:
    # 'A', 'B', 'C', 'D', and 'F' can be replaced with 5 features:
    # 'is_grade_A', 'is_grade_B', ..., and 'is_grade_F'.
    #
    # For the label, you can replace them with 0, 1, 2, 3, ...
    #
    # Returns
    # -----------
    # dataset : np.array
    #     2D array containing features and label.
    #
    # """
    # TODO: 1
    # raise NotImplementedError


def train(features, labels):
    tree_kun = DecisionTreeClassifier()
    tree_kun.fit(features, labels)
    return tree_kun

    # """
    # Return a decision tree model after "training".
    #
    # For more information on how to use Decision Tree, please visit
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    #
    # Returns
    # ---------
    # tree : sklearn.tree.DecisionTreeClassifier
    #
    # """
    # TODO 2:
    # raise NotImplementedError


def predict(model, features):
    return model.predict(features)
    # """
    # Return the prediction result.
    #
    # Parameters
    # --------
    # model : sklearn.tree.DecisionTreeClassifier
    # features : np.array
    #
    # Returns
    # -------
    # predictions : np.array
    # """
    # TODO 3:
    # raise NotImplementedError


def evaluate(labels, predictions, label_types):
    return confusion_matrix(labels, predictions, label_types)
    # """
    # Return the confusion matrix.
    #
    # Parameters
    # ---------
    # labels : np.array
    # predictions : np.array
    # label_types : np.array
    #     This is a list that keeps unique values of the labels, so that
    #     the confusion matrix has the same order.
    #
    # Returns
    # ---------
    # confusion_matrix : np.array
    #     The array should have the shape of [# of classes, # of classes].
    #     Number of class is the length of `label_types`.
    #
    # """
    # # TODO 4:
    # # You can use a library if you can find it.
    # raise NotImplementedError


if __name__ == "__main__":
    data = load_data()
    all_labels = data[:, -1]
    all_labels = set(all_labels)
    all_labels = np.array([v for v in all_labels])
    train_data, test_data = train_test_split(data, test_size=0.3)
    features = train_data[:, :-1]
    labels = train_data[:, -1]
    tree = train(features, labels)
    predictions = predict(tree, features)
    print('Training Confusion Matrix:')
    print(evaluate(labels, predictions, all_labels))
    predictions = predict(tree, test_data[:, :-1])
    print('Testing Confusion Matrix:')
    print(evaluate(test_data[:, -1], predictions, all_labels))
