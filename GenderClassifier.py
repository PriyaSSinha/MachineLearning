from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors

#training data [height,weight,shoe size]
X=[[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],[177,70,40],[159,55,37],[171,75,42],[181,85,43]]
Y = ['male', 'female', 'female', 'female', 'male', 'male','male', 'female', 'male', 'female', 'male']

clf=tree.DecisionTreeClassifier()

clf1 = svm.SVC()    #Support Vector classifier

clf2 = GaussianNB()   #Naive Bayes

clf3 = neighbors.KNeighborsClassifier() #K neighbors classifier

clf=clf.fit(X,Y)

clf1 = clf1.fit(X,Y)


clf2 = clf2.fit(X,Y)


clf3 = clf3.fit(X,Y)
 #test data

test=[[190,70,43],[170,43,38],[179,90,40]]
test1=['male','female','male']

predict=clf.predict(test)

prediction1 = clf1.predict(test)

prediction2 = clf2.predict(test)

prediction3 = clf3.predict(test)

print(predict)


print("Prediction for SVM : ",prediction1)

print("Accuracy for SVM : ",accuracy_score(test1,prediction1))

print("Prediction for Naive Bayes : ",prediction2)

print("Accuracy for Naive Bayes : ",accuracy_score(test1,prediction2))

print("Prediction for K neighbors : ",prediction3)

print("Accuracy for K neighbors : ",accuracy_score(test1,prediction3))

p=input("")