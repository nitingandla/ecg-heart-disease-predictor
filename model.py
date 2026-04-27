import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

X_dummy = np.random.rand(200, 300)   
y_dummy = np.random.randint(0, 2, 200)

pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_dummy)

model = LogisticRegression()
model.fit(X_reduced, y_dummy)

def predict(signals):
    
    features = np.array(signals).flatten().reshape(1, -1)

    features = np.resize(features, (1, 300))

    reduced = pca.transform(features)
    pred = model.predict(reduced)

    return pred[0]