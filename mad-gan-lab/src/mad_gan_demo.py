import numpy as np
from sklearn.preprocessing import StandardScaler

def synth_data(n=2000, d=8, anom_ratio=0.05, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, d))
    k = int(n * anom_ratio)
    X[:k] += rng.normal(6, 1.0, size=(k, d))  # simple anomalies
    y = np.zeros(n, dtype=int); y[:k] = 1
    return X, y

def main():
    X, y = synth_data()
    X = StandardScaler().fit_transform(X)
    print("Data ready:", X.shape, "Anomalies:", y.sum())
    print("TODO: plug in your MAD-GAN model here.")

if __name__ == "__main__":
    main()
