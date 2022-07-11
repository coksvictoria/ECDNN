import numpy as np
import scipy

from sklearn.base import BaseEstimator, ClassifierMixin

class KCDNNClassifier(BaseEstimator, ClassifierMixin):

  def __init__(self,n_neighbors=20,distance_metric='euclidean'):
      self.n_neighbors = n_neighbors
      self.distance_metric=distance_metric

  def fit(self, X, y):
      """Fit the k-nearest neighbors classifier from the training dataset.
      Parameters
      ----------
      X : {array-like, sparse matrix} of shape (n_samples, n_features)
          Training data.
      y : {array-like, sparse matrix} of shape (n_samples,)
          Target values.
      Returns
      -------
      self : KCDNNClassifier
          The fitted k- Centroid Displacement based nearest neighbors classifier.
      """
      self.X_ = X
      self.y_ = y

      self.fitted_ = True
      # Return the classifier
      return self

  def predict(self, X):

    if self.fitted_ == None:
		    raise Exception('predict() called before fit()')
   
    else:
      input_dim=X.shape[1]
      #calculate distance
      d=scipy.spatial.distance.cdist(X,self.X_,self.distance_metric)
      #get k lowest distance and save to Sx
      indexes=np.argsort(d)[:,:self.n_neighbors] # return k indexes of lowest value in d

      y_pred=[] ##set y_predict list
      for n,index in enumerate(indexes): ##looping through k indexes over the whole test dataset
        Sx = dict()
        for idx in range(self.n_neighbors):
          key = index[idx]
          if self.y_[key] in Sx:
            Sx[self.y_[key]].append(self.X_[key])
          else:
            Sx[self.y_[key]] = []
            Sx[self.y_[key]].append(self.X_[key])

        #calculate current centroids within training dataset
        px = dict()
        for key in Sx:
          sum_item = np.zeros(input_dim)
          for i in range(len(Sx[key])):
            sum_item += Sx[key][i]

          px_item = sum_item/len(Sx[key])

          px[key] = px_item

        #calculate new centroid by adding new test data
        qx = dict()
        for key in Sx:
          sum_item = np.zeros(input_dim)
          for i in range(len(Sx[key])):
            sum_item+=Sx[key][i]
          sum_item += X[n]
          qx_item = sum_item/(len(Sx[key]) + 1)
          qx[key] = qx_item

        #calculate displacement
        theta = dict()
        for key in px:
          if key in qx:
            theta[key] = np.linalg.norm(px[key] - qx[key])

        label=min(theta, key=theta.get)
        y_pred.append(label)
      return np.array(y_pred)
