import numpy as np

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    n_samples, n_features = X.shape
    
    # Initialize parameters
    w = np.zeros(n_features)
    b = 0
    
    for _ in range(steps):
        # Linear combination
        z = np.dot(X, w) + b
        
        # Prediction
        p = _sigmoid(z)
        
        # Gradients
        dw = (1 / n_samples) * np.dot(X.T, (p - y))
        db = (1 / n_samples) * np.sum(p - y)
        
        # Update
        w -= lr * dw
        b -= lr * db
    
    return w, b