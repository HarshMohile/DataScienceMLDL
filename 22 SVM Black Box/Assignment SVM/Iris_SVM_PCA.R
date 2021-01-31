 from sklearn.decomposition import PCA
 from sklearn.datasets import load_iris
 from sklearn import svm
 from sklearn import cross_validation
 import pylab as pl
 import numpy as np
 iris = load_iris()
 X_train, X_test, y_train, y_test =   cross_validation.train_test_split(iris.data,   iris.target, test_size=0.10, random_state=111)
 pca = PCA(n_components=2).fit(X_train)
 pca_2d = pca.transform(X_train)
 svmClassifier_2d =   svm.LinearSVC(random_state=111).fit(   pca_2d, y_train)
 for i in range(0, pca_2d.shape[0]):
   if y_train[i] == 0:
    c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    s=50,marker='+')
   elif y_train[i] == 1:
      c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    s=50,marker='o')
     elif y_train[i] == 2:
        c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',    s=50,marker='*')
       pl.legend([c1, c2, c3], ['Setosa', 'Versicolor',   'Virginica'])
       x_min, x_max = pca_2d[:, 0].min() - 1,   pca_2d[:,0].max() + 1
       y_min, y_max = pca_2d[:, 1].min() - 1,   pca_2d[:, 1].max() + 1
       xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
       Z = svmClassifier_2d.predict(np.c_[xx.ravel(),  yy.ravel()])
       Z = Z.reshape(xx.shape)
       pl.contour(xx, yy, Z)
       pl.title('Support Vector Machine Decision Surface')
       pl.axis('off')
       pl.show()