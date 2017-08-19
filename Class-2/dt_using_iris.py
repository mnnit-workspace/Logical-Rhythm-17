# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import datasets,tree,cross_validation
import graphviz as gz
import numpy as np 
import pydotplus
from sklearn.externals.six import StringIO

iris = datasets.load_iris()
#print(iris.target_names)
#print(iris.feature_names)
 
#print(iris.data[0])
#print(iris.target[0])
 
#for i in range(len(iris.target)):
#    print("Example %d: Label %s, Feature %s" % (i+1,iris.target[i],iris.data[i]))

test_idx = [0,50,100]

train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis=0)

test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))



dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
#graph = graphviz.Source(dot_data)  
#graph

gr = pydotplus.graph_from_dot_data(dot_data.getvalue())
gr.write_pdf("out.pdf")