"""
算例 2 验证 + 任务 4：空间任意方向杆单元
节点 1: (0, 0, 0), 节点 2: (1, 2, 2)
E = 210 GPa, A = 2.0e-4 m^2, de = [0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3]^T m
"""

import numpy as np
from truss3d import truss3d_element_stiffness, truss3d_element_stress

x1 = [0., 0., 0.]
x2 = [1., 2., 2.]
E  = 210e9
A  = 2.0e-4
de = [0., 0., 0., 1.0e-3, 2.0e-3, 2.0e-3]

L, (cx, cy, cz), Ke = truss3d_element_stiffness(x1, x2, E, A)
epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)
k = E * A / L

print('=' * 50)
print('算例 2 验证：空间任意方向杆单元')
print('=' * 50)

# 1. 单元长度
L_ok = np.isclose(L, 3.0, rtol=1e-10)
print(f'[{"PASS" if L_ok else "FAIL"}] 1. L = {L:.4f} m (期望 3.0 m)')

# 2. 方向余弦
dc_ok = (np.isclose(cx, 1/3, rtol=1e-10) and
         np.isclose(cy, 2/3, rtol=1e-10) and
         np.isclose(cz, 2/3, rtol=1e-10))
print(f'[{"PASS" if dc_ok else "FAIL"}] 2. (cx,cy,cz) = ({cx:.6f}, {cy:.6f}, {cz:.6f}) (期望 1/3, 2/3, 2/3)')

# 3. Ke 对称性
sym_ok = np.allclose(Ke, Ke.T, rtol=1e-10)
print(f'[{"PASS" if sym_ok else "FAIL"}] 3. Ke 对称性 (max|Ke-Ke^T| = {np.max(np.abs(Ke - Ke.T)):.2e})')

# 4. 刚体平移不产生内力
rigid_tests = [
    [1., 0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 1., 0.],
    [0., 0., 1., 0., 0., 1.],
    [2., 3., 5., 2., 3., 5.],
    [0.5, -1.2, 3.7, 0.5, -1.2, 3.7],
]
all_rigid_ok = True
for i, dr in enumerate(rigid_tests):
    Fe_r = Ke @ np.array(dr, dtype=float)
    ok = np.allclose(Fe_r, 0.0, atol=1e-8)
    if not ok: all_rigid_ok = False
    print(f'[{"PASS" if ok else "FAIL"}] 4.{i+1} 刚体平移 {dr[:3]} -> max|Fe| = {np.max(np.abs(Fe_r)):.1e}')
print(f'[{"PASS" if all_rigid_ok else "FAIL"}] 4. 刚体平移不产生内力')

# 5. 特征值分析
eigvals = np.linalg.eigvalsh(Ke)
max_eig = np.max(np.abs(eigvals))
tol = max(max_eig * 1e-12, 1e-8)
eigs_nonneg = np.all(eigvals >= -tol)
n_zero = np.sum(np.abs(eigvals) < tol)
print(f'[{"PASS" if eigs_nonneg else "FAIL"}] 5. 特征值非负 ({n_zero} 个零特征值, 秩=1, 奇异)')

# 6. 轴向应变
eps_ok = np.isclose(epsilon, 1.0e-3, rtol=1e-10)
print(f'[{"PASS" if eps_ok else "FAIL"}] 6. epsilon = {epsilon:.4e} (期望 1.0e-3)')

# 7. 轴向应力
sigma_ok = np.isclose(sigma, 210e6, rtol=1e-10)
print(f'[{"PASS" if sigma_ok else "FAIL"}] 7. sigma = {sigma/1e6:.2f} MPa (期望 210 MPa)')

# 8. 轴力
N_ok = np.isclose(N, 4.2e4, rtol=1e-10)
print(f'[{"PASS" if N_ok else "FAIL"}] 8. N = {N/1e3:.2f} kN (期望 42 kN)')

# 汇总
all_checks = [L_ok, dc_ok, sym_ok, all_rigid_ok, eigs_nonneg, eps_ok, sigma_ok, N_ok]
print(f'\n{"=" * 50}')
print(f'结果: {sum(all_checks)}/8 通过')
if all(all_checks):
    print('算例 2 全部验证通过')

# ============================================================
# 任务 4：刚度矩阵物理意义验证 (详细解释见 explanation.docx)
# ============================================================
print(f'\n{"=" * 50}')
print('任务 4：刚度矩阵物理意义验证')
print('=' * 50)

dof_names = ['u1', 'v1', 'w1', 'u2', 'v2', 'w2']
all_match = True
for j in range(6):
    de_t = np.zeros(6)
    de_t[j] = 1.0
    Fe_t = Ke @ de_t
    col_match = np.allclose(Fe_t, Ke[:, j], rtol=1e-12)
    if not col_match: all_match = False
    print(f'[{"PASS" if col_match else "FAIL"}] j={j} ({dof_names[j]}): Fe == Ke[:,{j}]')

print(f'\n[{"PASS" if all_match else "FAIL"}] 验证结论: Fe = Ke * e_j = Ke 的第 j 列')
print('物理意义: k_ij = 第 j 自由度单位位移引起的第 i 自由度约束力')
print('(详细解释见 explanation.docx)')
