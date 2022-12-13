from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree

class class_ExtraTreesClassifier:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        self.X = data
        self.y = label

    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)

    def model(self):
        # print("in model X :", self.X)
        # print(len(self.X))
        self.model = ExtraTreesClassifier(n_estimators=100, random_state=0)
        self.model.fit(np.array(self.X), self.y)

    def predict(self, data_testing):
        # print("prediksi : ", self.model.predict(data_testing))
        # print(data_testing)
        return self.model.predict(data_testing)

class class_RandomForest:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        print("Random forest classifier")
        self.X = data
        self.y = label
    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)
    def model(self):
        # print("in model X :", self.X)
        # print(len(self.X))
        self.model = RandomForestClassifier(n_estimators=10)
        self.model.fit(np.array(self.X), self.y)
    def predict(self, data_testing):
        # print("prediksi : ", self.model.predict(data_testing))
        #print(data_testing)
        return self.model.predict(data_testing)

class class_DecisionTree:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        print("Decession Tree Classification")
        self.X = data
        self.y = label

    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)

    def model(self):
        # print("in model X :", self.X)
        # print(len(self.X))
        self.model = tree.DecisionTreeClassifier()
        self.model.fit(np.array(self.X), self.y)

    def predict(self, data_testing):
        # print("prediksi : ", self.model.predict(data_testing))
        #print(data_testing)
        return self.model.predict(data_testing)

class class_MLP:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        print("MLP Classifier")
        self.X = data
        self.y = label

    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)

    def model(self):
        # print("in model X :", self.X)
        # print(len(self.X))
        self.model = MLPClassifier(hidden_layer_sizes=20,batch_size=10,max_iter=1000, random_state=1)
        self.model.fit(np.array(self.X), self.y)

    def predict(self, data_testing):
        # print("prediksi : ", self.model.predict(data_testing))
        # print(data_testing)
        return self.model.predict(data_testing)

class class_SVM:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        self.X = data
        self.y = label

    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)

    def model(self):
        # print("in model X :", self.X)
        # print(len(self.X))
        self.model = svm.SVR()
        self.model.fit(np.array(self.X), self.y)

    def predict(self, data_testing):
        # print("prediksi : ", self.model.predict(data_testing))
        # print(data_testing)
        return self.model.predict(data_testing)

class class_adaboostSVM:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        print("ADABOOST-SVC Classifier")
        self.X = data
        self.y = label

    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)

    def model(self):
        # print("in model X :", self.X)
        # print(len(self.X))
        svc=SVC(probability=True, kernel='rbf')
        self.model =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)
        self.model.fit(np.array(self.X), self.y)

    def predict(self, data_testing):
        # print("prediksi : ", self.model.predict(data_testing))
        # print(data_testing)
        return self.model.predict(data_testing)

class classLinearRegression:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        print("Linear Regression")
        self.X = data
        self.y = label

    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)

    def model(self):
        # print("in model X :", self.X)
        # print(len(self.X))
        self.model=LogisticRegression()
        self.model.fit(np.array(self.X), self.y)

    def predict(self, data_testing):
        # print("prediksi : ", self.model.predict(data_testing))
        # print(data_testing)
        return self.model.predict(data_testing)

class class_NB:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        self.X = data
        self.y = label

    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)

    def model(self):
        # print("in model X :", self.X)
        # print(len(self.X))
        self.model = GaussianNB()
        self.model.fit(np.array(self.X), self.y)

    def predict(self, data_testing):
        # print("prediksi : ", self.model.predict(data_testing))
        return self.model.predict(data_testing)


class class_KNN:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        self.X = data
        self.y = label

    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)

    def model(self,n_neighbors):
        # print("in model X :", self.X)
        # print(len(self.X))
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(np.array(self.X), self.y)

    def predict(self, data_testing):
        # print("prediksi : ", self.model.predict(data_testing))
        return self.model.predict(data_testing)