# Support vector machine (SVM) used as a classifier (Linear SVC)

import numpy as np  # numpy arrays
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style

style.use("ggplot")  # graph styling

# Take any entity with features:
# Two features, the x feature and the y feature:
x = [3.0, 3.9, 2.1, 9.3, 4.0, 5.9]
y = [1.6, 6.8, 1.1, 6.6, 5.6, 4.8]

plt.scatter(x, y)
plt.show()

X = np.array([[3.0, 1.6],
              [3.9, 6.8],
              [2.1, 1.1],
              [9.3, 6.6],
              [4.0, 5.6],
              [5.9, 4.8]])
