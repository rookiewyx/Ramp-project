"""评测用到的统计：准确率的置信区间、两次运行差异是否显著。

样本少的时候，“100% vs 96%”可能只是一两条样本的偶然波动。这里给出区间和检验，
避免把噪声当成改进或退步。
"""

from __future__ import annotations

from math import comb, sqrt


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间。比正态近似在小样本、接近 0% 或 100% 时可靠得多。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """配对比较两次运行（同一批用例）：b = 原来对现在错，c = 原来错现在对。

    只有结论发生翻转的用例携带信息；在“两次运行一样好”的假设下，翻转方向服从 p=0.5 的二项分布。
    返回双侧精确 p 值。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """两位标注人的一致性，扣除了“碰巧一致”的部分。1 = 完全一致，0 = 和随机标注差不多。

    经验上 > 0.8 算高度一致，0.6–0.8 算基本一致，< 0.6 说明标注标准本身有歧义，需要先统一口径。
    """
    n = len(a)
    if n == 0 or n != len(b):
        return None
    cats = set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 4)
