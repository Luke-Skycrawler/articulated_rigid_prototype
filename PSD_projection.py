import numpy as np


def project_PSD(A):
    lam, v = np.linalg.eigh(A)
    b = np.clip(lam, 0.0, None)
    # b = lam
    return v @ np.diag(b) @ v.T

if __name__ == "__main__":
    A = np.diag([1, -2, 3])
    A1 = project_PSD(A)
    B = np.ones((3, 1)) @ np.ones((1, 3))
    B1 = project_PSD(B)
    print(A1, B1)
