import numpy as np

def is_orthogonal(R, tol=1e-6):
    identity = np.identity(3)
    should_be_identity = np.dot(R.T, R)
    return np.allclose(should_be_identity, identity, atol=tol)

# 例として Camera1(id == 35とか) のRを確認
R1 = np.array([
    [0.03334258, -0.89149675,  0.45179843],
    [0.42895267,  0.42106224,  0.79919097],
    [-0.9027114,   0.16715306,  0.39644921]
])

print("Is R1 orthogonal?", is_orthogonal(R1))

det_R1 = np.linalg.det(R1)
print("Determinant of R1:", det_R1)
print("Is det(R1) approximately 1?", np.isclose(det_R1, 1.0, atol=1e-6))
