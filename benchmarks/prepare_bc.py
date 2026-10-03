import numpy as np
from sklearn.datasets import load_breast_cancer

d = load_breast_cancer()
X = d.data                       # 569 samples x 30 real features
y = (d.target == 0).astype(int)  # 1 = malignant, 0 = benign
np.save("features.npy", X)
np.save("labels.npy", y * 2)     # benchmark treats label >= 2 as positive
print(X.shape, "class counts (benign, malignant):", np.bincount(y))
