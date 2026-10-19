"""按龄折减增广的单元测试(容器内跑,镜像与 train.sh 相同)。

用法:cd YOPO && /opt/yopo-venv/bin/python policy/test_lag_augment.py
覆盖:投影式符号与数值、kill 分支、定量滞后、默认关时零随机数消耗、
lag_prob 统计。不装载数据集,只装配 mask 增广所需属性。
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from policy.yopo_dataset import YOPODataset, lag_displacement_px  # noqa: E402
from config.config import cfg                                      # noqa: E402


def bare_dataset(**over):
    d = YOPODataset.__new__(YOPODataset)
    d.mode = 'train'
    d.height, d.width = 96, 160
    d.target_radius = 0.15
    d.target_mask_min_px, d.target_mask_max_px = 3, 48
    d.target_mask_augment = True
    d.target_mask_jitter_px = 0.0            # 关抖动/缩放/dropout/假阳,隔离滞后项
    d.target_mask_scale_jitter = 0.0
    d.target_mask_dropout_prob = 0.0
    d.target_mask_false_positive_prob = 0.0
    d.target_mask_false_positive_max = 0
    d.target_mask_false_positive_min_px = 3
    d.target_mask_false_positive_max_px = 18
    d.target_mask_lag_enable = False
    d.target_mask_lag_prob = 0.5
    d.target_mask_lag_age_min_s = 0.05
    d.target_mask_lag_age_max_s = 0.7
    d.target_mask_lag_kill_prob = 0.12
    for k, v in over.items():
        setattr(d, k, v)
    return d


def bbox_center(mask):
    ys, xs = np.nonzero(mask)
    return ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0)


def main():
    fx = float(cfg["camera_fx"])
    fy = float(cfg["camera_fy"])
    depth, age = 2.0, 0.35
    vel_c = np.array([0.0, 1.0, 0.5], dtype=np.float32)   # 相机系:左 1 m/s、上 0.5 m/s
    uv = np.array([80.0, 48.0])
    duv = lag_displacement_px(vel_c, depth, age)

    # 1) 投影式:Δu=+fx·v_y·δ/d(向左移动的目标,旧位置在右侧)、Δv=+fy·v_z·δ/d
    assert np.allclose(duv, [fx * 1.0 * age / depth, fy * 0.5 * age / depth]), duv
    assert lag_displacement_px(vel_c, 0.05, age).tolist() == [0.0, 0.0]  # 深度无效→零位移

    # 2) kill 分支:kill_prob=1 → 可见目标整框丢弃(与"未检出"同表示)
    d = bare_dataset(target_mask_lag_enable=True, target_mask_lag_kill_prob=1.0)
    m = d._augment_mask(np.zeros((96, 160), np.float32), 1.0, uv, depth, vel_c)
    assert m.sum() == 0.0

    # 3) 滞后定量:prob=1、age 钉死 0.35 → mask 中心恰移 (Δu, Δv)
    d = bare_dataset(target_mask_lag_enable=True, target_mask_lag_kill_prob=0.0,
                     target_mask_lag_prob=1.0,
                     target_mask_lag_age_min_s=age, target_mask_lag_age_max_s=age)
    m0 = d._build_bbox_mask(1.0, uv, depth)
    m1 = d._augment_mask(m0.copy(), 1.0, uv, depth, vel_c)
    c0, c1 = bbox_center(m0), bbox_center(m1)
    assert abs((c1[0] - c0[0]) - duv[0]) <= 0.5 and abs((c1[1] - c0[1]) - duv[1]) <= 0.5, (c0, c1)

    # 4) 默认关:_sample_lag_px 零随机数消耗(与旧版同流),增广输出无滞后
    d = bare_dataset()
    np.random.seed(7)
    assert d._sample_lag_px(vel_c, depth).tolist() == [0.0, 0.0]
    a = np.random.rand()
    np.random.seed(7)
    b = np.random.rand()
    assert a == b
    np.random.seed(11)
    m2 = d._augment_mask(m0.copy(), 1.0, uv, depth, vel_c)
    assert abs(bbox_center(m2)[0] - 80.0) <= 0.5

    # 5) 统计:kill=0、prob=0.5 → 有滞后样本约占半(二项 ±5σ)
    d = bare_dataset(target_mask_lag_enable=True, target_mask_lag_kill_prob=0.0,
                     target_mask_lag_prob=0.5,
                     target_mask_lag_age_min_s=age, target_mask_lag_age_max_s=age)
    moved = 0
    for _ in range(2000):
        mm = d._augment_mask(m0.copy(), 1.0, uv, depth, vel_c)
        if abs(bbox_center(mm)[0] - 80.0) > 0.5:
            moved += 1
    assert 850 < moved < 1150, moved

    print("OK:投影式/kill/定量滞后/默认关零消耗/prob 统计 全过。")


if __name__ == "__main__":
    main()
