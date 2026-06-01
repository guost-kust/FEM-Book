"""
算例 1 验证：沿 x 轴的一维杆单元
节点 1: (0, 0, 0), 节点 2: (2, 0, 0)
E = 200 GPa, A = 1.0e-4 m^2, de = [0, 0, 0, 1.0e-3, 0, 0]^T m
"""

import numpy as np
from truss3d import truss3d_element_stiffness, truss3d_element_stress

x1 = [0., 0., 0.]
x2 = [2., 0., 0.]
E  = 200e9
A  = 1.0e-4
de = [0., 0., 0., 1.0e-3, 0., 0.]

L, (cx, cy, cz), Ke = truss3d_element_stiffness(x1, x2, E, A)
epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)
k = E * A / L

print('=' * 50)
print('算例 1 验证：沿 x 轴的一维杆单元')
print('=' * 50)

# 1. 单元长度
L_ok = np.isclose(L, 2.0, rtol=1e-10)
print(f'[{"PASS" if L_ok else "FAIL"}] 1. L = {L:.4f} m (期望 2.0 m)')

# 2. 方向余弦
dc_ok = (np.isclose(cx, 1., rtol=1e-10) and
         np.isclose(cy, 0., rtol=1e-10, atol=1e-15) and
         np.isclose(cz, 0., rtol=1e-10, atol=1e-15))
print(f'[{"PASS" if dc_ok else "FAIL"}] 2. (cx,cy,cz) = ({cx:.6f}, {cy:.6f}, {cz:.6f}) (期望 1, 0, 0)')

# 3. Ke 退化为一维杆单元形式
non_x = [1, 2, 4, 5]
zero_ok = np.allclose(Ke[non_x, :], 0, atol=1e-15) and np.allclose(Ke[:, non_x], 0, atol=1e-15)
sub_ok = np.allclose(Ke[np.ix_([0, 3], [0, 3])], k * np.array([[1., -1.], [-1., 1.]]), rtol=1e-10)
k_ok = np.isclose(Ke[0, 0], k, rtol=1e-10)
check3_ok = zero_ok and sub_ok and k_ok
print(f'[{"PASS" if check3_ok else "FAIL"}] 3. Ke 退化为一维杆单元形式 (k = {k:.2f} N/m)')

# 4. 轴向应变
eps_ok = np.isclose(epsilon, 5.0e-4, rtol=1e-10)
print(f'[{"PASS" if eps_ok else "FAIL"}] 4. epsilon = {epsilon:.4e} (期望 5.0e-4)')

# 5. 轴向应力
sigma_ok = np.isclose(sigma, 100e6, rtol=1e-10)
print(f'[{"PASS" if sigma_ok else "FAIL"}] 5. sigma = {sigma/1e6:.2f} MPa (期望 100 MPa)')

# 6. 轴力
N_ok = np.isclose(N, 1.0e4, rtol=1e-10)
print(f'[{"PASS" if N_ok else "FAIL"}] 6. N = {N/1e3:.2f} kN (期望 10 kN)')

# 汇总
all_checks = [L_ok, dc_ok, check3_ok, eps_ok, sigma_ok, N_ok]
print(f'\n{"=" * 50}')
print(f'结果: {sum(all_checks)}/6 通过')
if all(all_checks):
    print('算例 1 全部验证通过')
