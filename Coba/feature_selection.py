from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, chi2, \
    f_classif, f_regression, RFE, SelectFromModel
import numpy as np
from sklearn.svm import SVC

class feature_selection:
    X = []
    y = []

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def linearSVC(self):
        print("feature selection: linear SVC")
        lsvc = LinearSVC(
            C=0.01, penalty="l2", dual=False).fit(self.X, self.y)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(self.X)
        print(X_new.shape)
        return X_new

    def tree_based(self):
        print("feature selection: tree based")
        clf = ExtraTreesClassifier(n_estimators=30,criterion="entropy")
        clf = clf.fit(self.X, self.y)
        # print(clf.feature_importances_)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(self.X)
        print(X_new.shape)
        return X_new

    def pearson_correletion(self, threshold):
        print("feature selection: pearson correlation")
        # print("len X ",len(self.X[:,0]))
        # print("len y ",len(self.y))
        person=[]
        for i in range(len(self.X[0])):
            corr, _ = pearsonr(self.X[:,i], self.y)
            person.append(corr)
        # print(person)
        th = threshold
        ind = []
        newX = []
        for i in range(len(person)):
            if person[i]>th or person[i]<-th:
                ind.append(i)
                x = []
                for j in range(len(self.X)):
                    x.append(self.X[j][i])
                newX.append(x)
        # print(np.array(newX))
        # print(ind)
        return np.array(newX).transpose()

    def selectKbest_Anova(self, nfeature):
        print("select kbest - Anova")
        fvalue_Best = SelectKBest(f_classif, k=nfeature)
        X_kbest = fvalue_Best.fit_transform(self.X, self.y)
        return X_kbest

    def selectKbest_Chi2(self, nfeature):
        print("select kbest - chi square")
        # print(min(self.X[1]))
        fvalue_Best = SelectKBest(chi2, k=nfeature)
        X_kbest = fvalue_Best.fit_transform(self.X, self.y)
        # print(X_kbest)
        return X_kbest

    def selectKbest_regression(self, nfeature):
        print("select Kbest - regression")
        # print(min(self.X[1]))
        fvalue_Best = SelectKBest(f_regression, k=nfeature)
        X_kbest = fvalue_Best.fit_transform(self.X, self.y)
        # print(X_kbest)
        return X_kbest

    def RFE_SVC(self, k):
        print("recursive foward elimination - SVC")
        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
        rfe.fit(self.X, self.y)
        ranking = rfe.ranking_
        # print(self.X[0])
        # print(ranking)
        s = np.array(ranking)
        sort_index = np.argsort(ranking)
        # print(ranking)
        # print(sort_index)
        # print(np.sort(ranking))
        newX = []
        for i in range (len(self.X)):
            x = []
            for j in range(k):
                x.append(self.X[i][sort_index[j]])
            newX.append(x)
        newX = np.array(newX)
        return newX