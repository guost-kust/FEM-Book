import numpy as np


def truss3d_element_stiffness(x1, x2, E, A):
    """
    x1, x2: 节点坐标 [x, y, z]
    E: 弹性模量
    A: 截面积
    return: L, (cx, cy, cz), Ke
    """
    x1, x2 = np.array(x1, dtype=float), np.array(x2, dtype=float)
    d = x2 - x1
    L = np.linalg.norm(d)
    cx, cy, cz = d / L

    k = E * A / L

    Ke = k * np.array([
        [ cx*cx,  cx*cy,  cx*cz, -cx*cx, -cx*cy, -cx*cz],
        [ cx*cy,  cy*cy,  cy*cz, -cx*cy, -cy*cy, -cy*cz],
        [ cx*cz,  cy*cz,  cz*cz, -cx*cz, -cy*cz, -cz*cz],
        [-cx*cx, -cx*cy, -cx*cz,  cx*cx,  cx*cy,  cx*cz],
        [-cx*cy, -cy*cy, -cy*cz,  cx*cy,  cy*cy,  cy*cz],
        [-cx*cz, -cy*cz, -cz*cz,  cx*cz,  cy*cz,  cz*cz],
    ])

    return L, (cx, cy, cz), Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    de: [u1, v1, w1, u2, v2, w2]
    return: epsilon, sigma, N
    """
    x1, x2 = np.array(x1, dtype=float), np.array(x2, dtype=float)
    de = np.array(de, dtype=float)
    d = x2 - x1
    L = np.linalg.norm(d)
    cx, cy, cz = d / L

    B = np.array([[-cx, -cy, -cz, cx, cy, cz]]) / L

    epsilon = (B @ de)[0]
    sigma = E * epsilon
    N = A * sigma

    return epsilon, sigma, N


if __name__ == '__main__':
    # ============================================================
    # 算例 1
    # ============================================================
    print('=' * 50)
    print('算例 1')
    print('=' * 50)

    x1 = [0., 0., 0.]
    x2 = [2., 0., 0.]
    E  = 200e9       # 200 GPa
    A  = 1.0e-4
    de = [0., 0., 0., 1.0e-3, 0., 0.]

    L, (cx, cy, cz), Ke = truss3d_element_stiffness(x1, x2, E, A)
    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)

    print(f'杆长 L = {L:.4f} m')
    print(f'方向余弦 (cx, cy, cz) = ({cx:.6f}, {cy:.6f}, {cz:.6f})')
    print('刚度矩阵 Ke (6x6):')
    np.set_printoptions(precision=3, suppress=True)
    print(Ke)
    print(f'应变 epsilon = {epsilon:.4e}')
    print(f'应力 sigma = {sigma/1e6:.2f} MPa')
    print(f'轴力 N     = {N/1e3:.2f} kN')

    # ============================================================
    # 算例 2
    # ============================================================
    print()
    print('=' * 50)
    print('算例 2')
    print('=' * 50)

    x1 = [0., 0., 0.]
    x2 = [1., 2., 2.]
    E  = 210e9       # 210 GPa
    A  = 2.0e-4
    de = [0., 0., 0., 1.0e-3, 2.0e-3, 2.0e-3]

    L, (cx, cy, cz), Ke = truss3d_element_stiffness(x1, x2, E, A)
    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)

    print(f'杆长 L = {L:.4f} m')
    print(f'方向余弦 (cx, cy, cz) = ({cx:.6f}, {cy:.6f}, {cz:.6f})')
    print('刚度矩阵 Ke (6x6):')
    print(Ke)
    print(f'应变 epsilon = {epsilon:.4e}')
    print(f'应力 sigma = {sigma/1e6:.2f} MPa')
    print(f'轴力 N     = {N/1e3:.2f} kN')
