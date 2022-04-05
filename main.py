# Support vector machine (SVM) used as a classifier (SVC)

import numpy as np  # numpy arrays
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style

style.use("ggplot")  # graph styling

# Take any entity with features:
# Two features, the x feature and the y feature:
x = [1.0, 5.0, 1.5, 8.0, 1.0, 9.0]
y = [2.0, 8.0, 1.8, 8.0, 0.6, 11.0]

plt.scatter(x, y)
plt.show()
