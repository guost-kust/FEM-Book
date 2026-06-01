import numpy as np


def truss2d_element_stiffness(x1, x2, E, A):
    """
    x1, x2: 节点坐标 [x, y]
    E: 弹性模量
    A: 截面积
    return: L, (cx, cy), Ke (4x4)
    """
    x1, x2 = np.array(x1, dtype=float), np.array(x2, dtype=float)
    d = x2 - x1
    L = np.linalg.norm(d)
    cx, cy = d / L

    k = E * A / L

    Ke = k * np.array([
        [ cx*cx,  cx*cy, -cx*cx, -cx*cy],
        [ cx*cy,  cy*cy, -cx*cy, -cy*cy],
        [-cx*cx, -cx*cy,  cx*cx,  cx*cy],
        [-cx*cy, -cy*cy,  cx*cy,  cy*cy],
    ])

    return L, (cx, cy), Ke


def truss2d_element_stress(x1, x2, E, A, de):
    """
    de: [u1, v1, u2, v2]
    return: epsilon, sigma, N
    """
    x1, x2 = np.array(x1, dtype=float), np.array(x2, dtype=float)
    de = np.array(de, dtype=float)
    d = x2 - x1
    L = np.linalg.norm(d)
    cx, cy = d / L

    B = np.array([[-cx, -cy, cx, cy]]) / L

    epsilon = (B @ de)[0]
    sigma = E * epsilon
    N = A * sigma

    return epsilon, sigma, N


if __name__ == '__main__':
    # ============================================================
    # 算例 1：水平杆单元
    # ============================================================
    print('=' * 50)
    print('算例 1：水平杆单元')
    print('=' * 50)

    x1 = [0., 0.]
    x2 = [2., 0.]
    E  = 200e9
    A  = 1.0e-4
    de = [0., 0., 1.0e-3, 0.]

    L, (cx, cy), Ke = truss2d_element_stiffness(x1, x2, E, A)
    epsilon, sigma, N = truss2d_element_stress(x1, x2, E, A, de)

    print(f'杆长 L = {L:.4f} m')
    print(f'方向余弦 (cx, cy) = ({cx:.6f}, {cy:.6f})')
    print('刚度矩阵 Ke (4x4):')
    np.set_printoptions(precision=3, suppress=True)
    print(Ke)
    print(f'应变 epsilon = {epsilon:.4e}')
    print(f'应力 sigma = {sigma/1e6:.2f} MPa')
    print(f'轴力 N     = {N/1e3:.2f} kN')

    # ============================================================
    # 算例 2：倾斜杆单元
    # ============================================================
    print()
    print('=' * 50)
    print('算例 2：倾斜杆单元')
    print('=' * 50)

    x1 = [0., 0.]
    x2 = [3., 4.]
    E  = 210e9
    A  = 2.0e-4
    de = [0., 0., 3.0e-3, 4.0e-3]

    L, (cx, cy), Ke = truss2d_element_stiffness(x1, x2, E, A)
    epsilon, sigma, N = truss2d_element_stress(x1, x2, E, A, de)

    print(f'杆长 L = {L:.4f} m')
    print(f'方向余弦 (cx, cy) = ({cx:.6f}, {cy:.6f})')
    print('刚度矩阵 Ke (4x4):')
    print(Ke)
    print(f'应变 epsilon = {epsilon:.4e}')
    print(f'应力 sigma = {sigma/1e6:.2f} MPa')
    print(f'轴力 N     = {N/1e3:.2f} kN')
