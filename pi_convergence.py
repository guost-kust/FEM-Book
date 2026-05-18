import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 割圆术 + 外推法 (Extrapolation) 收敛性研究
# 课件核心逻辑:
#   1. π_n = n·sin(π/n) — 正多边形逼近 (h-加密, O(h²) 收敛)
#   2. 外推法: 利用已知误差展开 (仅含 h 的偶次幂) 消除低阶误差项
#      Richardson 外推: R_k = (4^k·R_{k-1}(h/2) - R_{k-1}(h)) / (4^k - 1)
#      等价于 Wynn-ε 算法的偶数列
#   3. 对每个 n, 使用尽可能多的已有数据做最高阶外推
# ============================================================

pi_true = 4.0 * np.arctan(1.0)  # true π

# 正多边形序列
N_vals = [4 * (2 ** i) for i in range(10)]  # n = 4, 8, 16, ..., 2048
pi_poly = np.array([n * np.sin(pi_true / n) for n in N_vals])
h_vals = 1.0 / np.array(N_vals, dtype=float)
err_poly = np.abs(pi_true - pi_poly)

# ============================================================
# 建立 Romberg 外推表
# R[0][i] = π_{n_i} (原始多边形逼近)
# R[k][i] = (4^k * R[k-1][i+1] - R[k-1][i]) / (4^k - 1)
# R[k] 误差为 O(h^{2k+2}), 理论斜率 = 2k+2
# ============================================================
num_pts = len(N_vals)
R = [pi_poly]
for k in range(1, num_pts):
    c = 4.0 ** k
    row = []
    for i in range(num_pts - k):
        val = (c * R[k - 1][i + 1] - R[k - 1][i]) / (c - 1.0)
        row.append(val)
    R.append(np.array(row))

# ============================================================
# 递进式外推: 对每个 n 使用最大可用阶数
# n=4  (i=0): 无外推 (仅有 1 个数据) → 占位值
# n=8  (i=1): R1[0] → 1 次外推, 使用 n=4,8
# n=16 (i=2): R2[0] → 2 次外推, 使用 n=4,8,16
# ...
# n=2048 (i=9): R9[0] → 9 次外推, 使用全部 10 个数据
# ============================================================
print(f"{'n':>5}  {'pi_n (多边形)':>18}  {'pi_extrap (外推)':>22}  "
      f"{'外推阶数':>8}  {'log10(err_ori)':>14}")
print("-" * 85)

extrap_vals = []   # 外推值
extrap_errs = []   # 外推误差
extrap_h = []      # 对应 h

for i in range(num_pts):
    n = N_vals[i]
    h = h_vals[i]
    err_o = err_poly[i]

    if i == 0:
        # n=4: 无外推可用
        extrap_vals.append(0.0)
        extrap_errs.append(-1.0)  # 占位
        extrap_h.append(h)
        print(f"{n:5d}  {pi_poly[i]:18.12f}  {'---':>22}  {'---':>8}  "
              f"{np.log10(err_o):14.6f}")
    else:
        k = i                     # 外推阶数 = i
        val = R[k][0]             # R_k[0] 使用 n=4, 8, ..., 4·2^i
        err = abs(pi_true - val)
        extrap_vals.append(val)
        extrap_errs.append(err)
        extrap_h.append(h)
        print(f"{n:5d}  {pi_poly[i]:18.12f}  {val:22.16f}  {k:>8}  "
              f"{np.log10(err_o):14.6f}")

# ============================================================
# 保存数据文件
# ============================================================
with open("pi_convergence_data.txt", "w") as f:
    f.write(f"{'n':>6}  {'h=1/n':>18}  {'error_original':>18}  "
            f"{'error_extrapolated':>18}\n")
    for i in range(num_pts):
        f.write(f"{N_vals[i]:6d}  {h_vals[i]:18.10E}  {err_poly[i]:18.10E}  "
                f"{extrap_errs[i]:18.10E}\n")

print("\n数据已写入 pi_convergence_data.txt")

# ============================================================
# 计算收敛阶 (斜率)
# ============================================================


def calc_slope(x, y):
    """log-log 线性拟合"""
    coeff = np.polyfit(np.log10(x), np.log10(y), 1)
    return coeff[0]


# 原始误差 vs h
slope_ori = calc_slope(h_vals, err_poly)

# 外推误差 vs h (跳过占位的 n=4)
valid = np.array([e for e in extrap_errs[1:] if e > 1e-16])
valid_h = np.array([extrap_h[i] for i in range(1, len(extrap_h))
                    if extrap_errs[i] > 1e-16])

# 仅用误差显著高于机器精度的点算斜率
slope_extrap = calc_slope(valid_h, valid)

print(f"\n原始逼近斜率: {slope_ori:.2f} (理论值 2.00)")
print(f"外推收敛斜率: {slope_extrap:.2f} (课件目标 ~9.76)")

# ============================================================
# 绘图
# ============================================================
plt.figure(figsize=(9, 7.5))

# 蓝色线：原始多边形逼近
plt.loglog(h_vals, err_poly, "b-", marker="v", markersize=9,
           linewidth=1.5, label=f"Original  slope: {slope_ori:.2f}")

# 红色/橙色线：外推法
plt.loglog(valid_h, valid, "r-", marker="^", markersize=9,
           linewidth=1.8, label=f"Extrapolation  slope: {slope_extrap:.2f}")

# 标注斜率
plt.text(0.04, 0.78, f"slope: {slope_ori:.2f}", color="blue",
         transform=plt.gca().transAxes, fontsize=18, fontweight="bold")
plt.text(0.35, 0.12, f"slope: {slope_extrap:.2f}", color="red",
         transform=plt.gca().transAxes, fontsize=18, fontweight="bold")

plt.xlabel("h = 1/n", fontsize=15)
plt.ylabel(r"$e_n = |\pi - \pi_n|$", fontsize=15)
plt.xlim(1e-3, 1e0)
plt.ylim(1e-16, 1e5)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=13, loc="lower right")
plt.title("Convergence Study of π Approximation (Liu Hui's Method + Extrapolation)",
          fontsize=15)

plt.tight_layout()
plt.savefig("pi_convergence_plot.png", dpi=300)
plt.show()
