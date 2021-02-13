import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_pts(csv_name):
    data = np.asarray(pd.read_csv(csv_name, header=None))
    X = data[:, 0:2]
    y = data[:, 2]

    plt.scatter(X[np.argwhere(y == 0).flatten(), 0],
                X[np.argwhere(y == 0).flatten(), 1],
                s=50,
                color='blue',
                edgecolor='k')
    plt.scatter(X[np.argwhere(y == 1).flatten(), 0],
                X[np.argwhere(y == 1).flatten(), 1],
                s=50,
                color='red',
                edgecolor='k')

    plt.xlim(-2.05, 2.05)
    plt.ylim(-2.05, 2.05)
    plt.grid(False)
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off'
    )
    return X, y


def plot_model(X, y, clf):
    plt.scatter(X[np.argwhere(y == 0).flatten(), 0],
                X[np.argwhere(y == 0).flatten(), 1],
                s=50,
                color='blue',
                edgecolor='k')
    plt.scatter(X[np.argwhere(y == 1).flatten(), 0],
                X[np.argwhere(y == 1).flatten(), 1],
                s=50,
                color='red',
                edgecolor='k')

    plt.xlim(-2.05, 2.05)
    plt.ylim(-2.05, 2.05)
    plt.grid(False)
    plt.tick_params(axis='x',
                    which='both',
                    bottom='off',
                    top='off')

    r = np.linspace(-2.1, 2.1, 300)
    s, t = np.meshgrid(r, r)
    s = np.reshape(s, (np.size(s), 1))
    t = np.reshape(t, (np.size(t), 1))
    h = np.concatenate((s, t), 1)

    z = clf.predict(h)

    s = s.reshape((np.size(r), np.size(r)))
    t = t.reshape((np.size(r), np.size(r)))
    z = z.reshape((np.size(r), np.size(r)))

    plt.contourf(s, t, z, colors=['blue', 'red'], alpha=0.2, levels=range(-1, 2))
    if len(np.unique(z)) > 1:
        plt.contour(s, t, z, colors='k', linewidths=2)
    plt.show()


X, y = load_pts('grid_search_lab_data.csv')
plt.show()

# Fixing a random seed
random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model (with default hyperparameters)
clf = DecisionTreeClassifier(random_state=42)

# Fit the model
clf.fit(X_train, y_train)

# Make predictions
train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)

plot_model(X, y, clf)
print('The Training F1 Score is', f1_score(train_predictions, y_train))
print('The Testing F1 Score is', f1_score(test_predictions, y_test))

parameters = {'max_depth': [2, 4, 6, 8, 10],
              'min_samples_leaf': [2, 4, 6, 8, 10],
              'min_samples_split': [2, 4, 6, 8, 10]}


def calculate_f1_score(parameters_to_test):
    # TODO: Make an fbeta_score scoring object.
    scorer = make_scorer(f1_score)

    # TODO: Perform grid search on the classifier using 'scorer' as the scoring method.
    grid_obj = GridSearchCV(clf, parameters_to_test, scoring=scorer)

    # TODO: Fit the grid search object to the training data and find the optimal parameters.
    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the estimator.
    best_clf = grid_fit.best_estimator_

    # Fit the new model.
    best_clf.fit(X_train, y_train)

    # Make predictions using the new model.
    best_train_predictions = best_clf.predict(X_train)
    best_test_predictions = best_clf.predict(X_test)

    # Calculate the f1_score the new model.
    print('The training F1 Score is', f1_score(best_train_predictions, y_train))
    print('The testing F1 Score is', f1_score(best_test_predictions, y_test))

    # Plot the new model.
    plot_model(X, y, best_clf)

    # Let's also explore what parameters ended up being used in the new model.
    print(best_clf)


# ----------------------------------------------------------- #

# Call the function.
calculate_f1_score(parameters)

parameters = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
calculate_f1_score(parameters)
