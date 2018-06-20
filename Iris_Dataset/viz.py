# Iris Dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set
#
# Youtube video: https://www.youtube.com/watch?v=tNa99PG8hR8&index=2&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal
#

from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()
'''
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])

# Iris Dataset
for i in range(len(iris.target)):
    print("Examples %d: lable %s, feature %s" % (i, iris.target[i], iris.data[i]))
'''

test_idx = [0, 50, 100]

# Training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Classifier
clf = tree.DecisionTreeClassifier()
clf.fit (train_data, train_target)

print(test_target)
print(clf.predict(test_data))

# Intall Graphviz:
# >> http://macappstore.org/graphviz/
# >> brew install graphviz

import graphviz

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

dot_data = tree.export_graphviz(clf, out_file=None,
feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph

print(test_data[1], test_target[1])
print(iris.feature_names, iris.target_names)


# End of file