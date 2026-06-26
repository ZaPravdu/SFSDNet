"""
LabeledSet ratio 核心逻辑测试（无需 GPU，纯 CPU 验证）。
覆盖：
  1. LabeledSet 的 ratio → per_scene_max 计算（启用/禁用/min1边界）
  2. _compute_delta_L 的过滤条件（ratio 开关语义）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import random
import torch
from model.labeled_set import LabeledSet

# ============================================================
# 1. 测试 LabeledSet ratio → per_scene_max 计算
# ============================================================
def test_labeled_set_ratio_to_max():
    """验证 ratio+scene_totals → 内部 per_scene_max 正确"""
    print("=== 测试 1: LabeledSet ratio → per_scene_max ===")

    # Case 1: ratio=0 → _per_scene_max 为空，永远返回 False
    ls = LabeledSet(ratio=0, scene_totals={'scene_8': 200})
    ls.add('scene_8/10_30')
    assert 'scene_8/10_30' not in ls, "ratio=0 时不添加任何样本"
    print("  ✓ ratio=0: 不添加任何样本")

    # Case 2: ratio=0.1, scene_8=200 → max(1, int(0.1*200)) = 20
    ls = LabeledSet(ratio=0.1, scene_totals={'scene_8': 200})
    for i in range(25):
        ls.add(f'scene_8/{i}_{i+1}')
    assert len(ls._ids) == 20, f"ratio=0.1 时 200 帧应选 20 个，实际 {len(ls._ids)}"
    assert ls._per_scene_max['scene_8'] == 20
    print(f"  ✓ ratio=0.1 × 200 → max = {ls._per_scene_max['scene_8']} (正确)")

    # Case 3: 极小场景，floor 后为 0，但 min=1
    ls = LabeledSet(ratio=0.01, scene_totals={'tiny_scene': 30})
    assert ls._per_scene_max['tiny_scene'] == max(1, int(0.01 * 30)), \
        f"min=1 边界: int(0.01*30)={int(0.01*30)}, max(1, ...)={max(1, int(0.01*30))}"
    ls.add('tiny_scene/1_5')
    assert 'tiny_scene/1_5' in ls, "极小场景至少能选 1 个"
    print(f"  ✓ ratio=0.01 × 30 → max = {ls._per_scene_max['tiny_scene']} (min=1 生效)")

    # Case 4: 多个场景独立计算
    ls = LabeledSet(ratio=0.5, scene_totals={'a': 10, 'b': 100})
    assert ls._per_scene_max['a'] == max(1, int(0.5 * 10))
    assert ls._per_scene_max['b'] == max(1, int(0.5 * 100))
    assert ls._per_scene_max['a'] == 5
    assert ls._per_scene_max['b'] == 50
    print(f"  ✓ 多场景独立: a={ls._per_scene_max['a']}, b={ls._per_scene_max['b']}")

    # Case 5: serialize/deserialize preserves per_scene_max
    ls = LabeledSet(ratio=0.1, scene_totals={'x': 100})
    for i in range(5):
        ls.add(f'x/{i}_{i+1}')
    data = ls.serialize()
    assert data['ratio'] == 0.1
    assert data['_per_scene_max']['x'] == 10
    ls2 = LabeledSet()
    ls2.deserialize(data)
    assert ls2.ratio == 0.1
    assert ls2._per_scene_max['x'] == 10
    assert 'x/0_1' in ls2
    print("  ✓ serialize/deserialize: ratio & per_scene_max 正确持久化")

# ============================================================
# 2. 测试 _compute_delta_L 过滤逻辑（ratio 开关语义）
# ============================================================
def test_delta_L_filtering():
    """模拟 _compute_delta_L 的循环过滤逻辑（不跑真实模型）"""
    print("=== 测试 2: ΔL 循环过滤逻辑 ===")

    n_total = 20
    gt_indices = set(random.sample(range(n_total), 5))

    # 场景 A: gt_ratios_per_scene=0（禁用）
    gt_ratios_per_scene = 0
    processed = []
    for i in range(n_total):
        if gt_ratios_per_scene:
            if i not in gt_indices:
                continue
            processed.append(i)
        else:
            processed.append(i)
    assert len(processed) == n_total, f"禁用时应处理 {n_total} 个，实际 {len(processed)}"
    print("  ✓ gt_ratios_per_scene=0: 处理全部 20 个样本")

    # 场景 B: gt_ratios_per_scene>0（启用）
    gt_ratios_per_scene = 0.1
    processed = []
    for i in range(n_total):
        if gt_ratios_per_scene:
            if i not in gt_indices:
                continue
            processed.append(i)
        else:
            processed.append(i)
    assert len(processed) == len(gt_indices), f"启用时应只处理 GT 样本 ({len(gt_indices)} 个)，实际 {len(processed)}"
    assert set(processed) == gt_indices, "处理的样本应是 GT 索引集合"
    print(f"  ✓ gt_ratios_per_scene>0: 只处理 {len(gt_indices)} 个 GT 样本")

    # 场景 C: 边界条件 — gt_ratios_per_scene=0
    gt_ratios_per_scene = 0
    processed = []
    for i in range(n_total):
        if gt_ratios_per_scene:
            if i not in gt_indices:
                continue
            processed.append(i)
        else:
            processed.append(i)
    assert len(processed) == n_total, f"禁用时处理全部，实际 {len(processed)}"
    print("  ✓ 边界: gt_ratios_per_scene=0 时处理全部样本")


if __name__ == '__main__':
    print("=" * 50)
    print("LabeledSet ratio 核心逻辑测试")
    print("=" * 50)
    test_labeled_set_ratio_to_max()
    test_delta_L_filtering()
    print("\n" + "=" * 50)
    print("全部测试通过 ✓")
    print("=" * 50)
