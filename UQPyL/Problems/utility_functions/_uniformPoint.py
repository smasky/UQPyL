import numpy as np

def uniformPoint(N, M):
    H1 = int(np.floor(np.sqrt(N)))
    H2 = int(np.ceil(N / H1))
    while H1 * H2 < N:
        if H1 < H2:
            H1 = H1 + 1
        else:
            H2 = H2 + 1
    W = np.zeros((H1 * H2, M))
    temp = np.zeros((1, M))
    for i in range(2, M + 1):
        temp[0, i - 1] = 1
        W[:, i - 1] = np.floor(np.linspace(temp[0, i - 1], H1, H1 * H2) / H1)
    W = (W[::-1, :] + 1) / (M - np.sum(W, axis=1, keepdims=True) + 1)
    return W[:N, :]