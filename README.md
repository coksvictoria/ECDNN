# Ensemble Centroid Displacement-based $k$-NN (ECDNN)

The CDNN algorithm is proposed in the paper [Robust Biometric Recognition From Palm Depth Images for Gloved Hands](https://ieeexplore.ieee.org/document/7161357). This paper presents a simple yet efficient variant of the CDNN algorithm, named Ensemble Centroid Displacement-based $k$-NN (ECDNN), which leverages the homogeneity of the nearest neighbors for each test instance. To the best of our knowledge, this is the first ensemble $k$-NN algorithm which incorporates the Data-centric Artificial Intelligence (AI) philosophy, where both algorithm and data are integral components for successful applications. The intuition behind the proposed algorithm is simple: we use simple algorithm for high confidence sample, where its nearest neighbors are homogeneous and hence the prediction is easy to make with high confidence. Otherwise, we use complex algorithm for low confidence sample, where intra-class distribution is overlapping and the in-between distance is small. Therefore, the proposed model aims to find a conditional feature space based on not only the distance metric, but also the homogeneity of the nearest neighbors to the test instance. By aggregating the benefits of both underlying models, the proposed algorithm is expected to improve the computational efficiency of the complex algorithm and also enhance the discriminative capability for classification.




The repositry includes:
- Native Python implementations of ECDNN alongside a flexible framework for adapting different distance metrics.
- Examples of using ECDNN
- A comparision between ECDNN and tradditional k-NN algorithm on some sample datasets
- A comparision of using different distance metrics with ECDNN

Please refer to example.ipynb for examples.

A sample result will look like this:
```
Testing with k = 21

---------------Digits dataset------------------
Loading data.....
Done loading data!

Number of classes: 10
Data dimension: 64
Number of training samples: 1437
Number of testing samples: 360

Predict time for ECDNN: 0.050s
Accuracy for ECDNN with k = 21: 0.992

Predict time for kNN with uniform weights: 0.025s
Accuracy for kNN with k = 21 and uniform weights: 0.978

Predict time for kNN with distance weights: 0.015s
Accuracy for kNN with k = 21 and distance weights: 0.983
```
