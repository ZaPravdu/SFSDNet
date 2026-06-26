"""
可视化函数测试脚本（纯 CPU，无需模型/GPU）。
覆盖：
  1. 正常 batch 数据（有 GT + Pred）
  2. 只有预测（无 GT）
  3. 单图输入 (3,H,W)
  4. 密度图全为零（边界情况）
  5. 验证输出是合法 PNG
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from model_assembler import visualize_density_debug


def is_valid_png(path):
    """检查文件是否为合法 PNG。"""
    if not os.path.isfile(path):
        return False, '文件不存在'
    if os.path.getsize(path) == 0:
        return False, '文件大小为 0'
    with open(path, 'rb') as f:
        header = f.read(8)
        if header != b'\x89PNG\r\n\x1a\n':
            return False, f'文件头不正确'
    return True, 'OK'


def make_img(b=1, h=288, w=384):
    """生成模拟归一化图片。b=1 时返回 (3,h,w)，否则 (b,3,h,w)。"""
    # 用固定种子确保可复现
    rng = torch.Generator().manual_seed(42)
    img = torch.randn(b, 3, h, w, generator=rng) * 0.5 + 0.4
    # 截断到合理归一化范围
    img = img.clamp(-2.5, 2.5)
    if b == 1:
        return img[0]
    return img


def make_dens(b=1, h=288, w=384, n_peaks=5, peak_val=0.05):
    """生成模拟密度图（几个高斯峰）。"""
    rng = np.random.default_rng(42)
    d = np.zeros((b, 1, h, w), dtype=np.float32)
    for i in range(b):
        for _ in range(n_peaks):
            cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
            xs = np.arange(w, dtype=np.float32).reshape(1, -1)
            ys = np.arange(h, dtype=np.float32).reshape(-1, 1)
            g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 20 ** 2))
            d[i, 0] += g * peak_val
    if b == 1:
        return torch.from_numpy(d[0:1])
    return torch.from_numpy(d)


def test_normal_batch():
    """测试 1：正常 batch (b=4)，同时有 GT 和 Pred。"""
    print('=== 测试 1: 正常 batch (b=4) ===')
    img = make_img(4)
    pred = {'global': make_dens(4), 'share': make_dens(4, n_peaks=3, peak_val=0.03),
            'in_out': make_dens(4, n_peaks=2, peak_val=0.02)}
    gt = {'global': make_dens(4, n_peaks=6, peak_val=0.06),
          'share': make_dens(4, n_peaks=4, peak_val=0.04),
          'in_out': make_dens(4, n_peaks=2, peak_val=0.02)}
    visualize_density_debug(
        img, pred, gt,
        save_path='/tmp/test_viz_normal.png',
        idx=2, info='scene_8/0001-0002',
    )
    valid, msg = is_valid_png('/tmp/test_viz_normal.png')
    assert valid, f'输出无效: {msg}'
    sz = os.path.getsize('/tmp/test_viz_normal.png')
    print(f'  ✓ 输出正常 ({sz} bytes)')


def test_pred_only():
    """测试 2：只有预测（无 GT），1 行布局。"""
    print('=== 测试 2: 仅预测（无 GT） ===')
    img = make_img(2)
    pred = {'global': make_dens(2), 'share': make_dens(2, n_peaks=3), 'in_out': make_dens(2)}
    visualize_density_debug(img, pred, save_path='/tmp/test_viz_pred_only.png')
    valid, msg = is_valid_png('/tmp/test_viz_pred_only.png')
    assert valid, f'输出无效: {msg}'
    print(f'  ✓ 输出正常 ({os.path.getsize("/tmp/test_viz_pred_only.png")} bytes)')


def test_single_image():
    """测试 3：单图 (3,H,W)，无 batch 维度。"""
    print('=== 测试 3: 单图 (3,H,W) ===')
    img = make_img(1)
    pred = {'global': make_dens(1), 'share': make_dens(1), 'in_out': make_dens(1)}
    visualize_density_debug(img, pred, pred, save_path='/tmp/test_viz_single.png')
    valid, msg = is_valid_png('/tmp/test_viz_single.png')
    assert valid, f'输出无效: {msg}'
    print(f'  ✓ 输出正常 ({os.path.getsize("/tmp/test_viz_single.png")} bytes)')


def test_all_zero_dens():
    """测试 4：密度图全零（边界情况）。"""
    print('=== 测试 4: 密度图全零 ===')
    img = make_img(2)
    d = torch.zeros(2, 1, 288, 384)
    zero = {'global': d, 'share': d.clone(), 'in_out': d.clone()}
    visualize_density_debug(img, zero, zero, idx=1, save_path='/tmp/test_viz_all_zero.png')
    valid, msg = is_valid_png('/tmp/test_viz_all_zero.png')
    assert valid, f'输出无效: {msg}'
    print(f'  ✓ 输出正常 ({os.path.getsize("/tmp/test_viz_all_zero.png")} bytes)')


def test_with_cfg_denormalize():
    """测试 5：传入 cfg_data 反归一化。"""
    print('=== 测试 5: 带 cfg_data 反归一化 ===')
    img = make_img(2)

    class MockCfg:
        MEAN_STD = ([0.4588, 0.4314, 0.4118], [0.2631, 0.2567, 0.2597])

    pred = {'global': make_dens(2), 'share': make_dens(2), 'in_out': make_dens(2)}
    visualize_density_debug(img, pred, pred, cfg_data=MockCfg(),
                            save_path='/tmp/test_viz_cfg.png', info='with_cfg')
    valid, msg = is_valid_png('/tmp/test_viz_cfg.png')
    assert valid, f'输出无效: {msg}'
    print(f'  ✓ 输出正常 ({os.path.getsize("/tmp/test_viz_cfg.png")} bytes)')


if __name__ == '__main__':
    print('=' * 55)
    print('可视化函数测试')
    print('=' * 55)
    test_normal_batch()
    test_pred_only()
    test_single_image()
    test_all_zero_dens()
    test_with_cfg_denormalize()
    print('\n' + '=' * 55)
    print('全部测试通过 ✓')
    print('=' * 55)
