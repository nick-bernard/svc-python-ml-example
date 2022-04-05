# Support vector machine (SVM) used as a classifier (Linear SVC)

import numpy as np  # numpy arrays
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style

style.use("ggplot")  # graph styling

# Take any entity with features:
# Two features, the x feature and the y feature:
x = [3.0, 3.9, 2.1, 5.3, 4.0, 3.9]
y = [1.6, 6.8, 1.1, 6.6, 5.6, 5.8]

plt.scatter(x, y)
plt.show()

X = np.array([[3.0, 1.6],
              [3.9, 6.8],
              [2.1, 1.1],
              [5.3, 6.6],
              [4.0, 5.6],
              [3.9, 5.8]])

# Binary classification scheme (classified based on "low" or "high")
y = [0, 1, 0, 1, 1, 1]

# Create our classifier
# C defined explicitly here, although default parameter is also 1.0, so this could have been left implicit
cls = svm.SVC(kernel='linear', C=1.0)

# fit features to their labels
cls.fit(X, y)

p = cls.predict([4, 6])
