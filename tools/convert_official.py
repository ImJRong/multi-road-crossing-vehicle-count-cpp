#!/usr/bin/env python3
"""按官方 rknn_model_zoo 示例转 YOLOv8（参数极简，输出保持 float32）。

官方脚本（examples/yolov8/python/convert.py）就三行：
    rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]],
                target_platform=platform)

我上一版多加了 quantized_dtype='w8a8'，正是它把输入输出压成 int8 的。

用法（WSL 里）：
    source ~/rknn_env.sh
    python ~/traffic_count/tools/convert_official.py
"""
import os
import sys

from rknn.api import RKNN

PROJ = os.path.expanduser("~/traffic_count")
ONNX = os.path.join(PROJ, "models/best_detrac_head2.onnx")
CALIB_DIR = "/mnt/d/rknn_calib"
DATASET = os.path.join(PROJ, "dataset_calib.txt")


def make_dataset():
    imgs = sorted(f for f in os.listdir(CALIB_DIR) if f.lower().endswith(".jpg"))
    with open(DATASET, "w", encoding="utf-8") as f:
        for name in imgs:
            f.write(os.path.join(CALIB_DIR, name) + "\n")
    print("校准图 %d 张" % len(imgs))


def build(out_path):
    print("\n" + "=" * 50)
    print("转换 -> %s" % os.path.basename(out_path))
    print("=" * 50)

    rknn = RKNN(verbose=False)
    # 官方示例就这么配，多余参数一律不加
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )

    print("加载 ONNX ...")
    if rknn.load_onnx(model=ONNX) != 0:
        raise RuntimeError("load_onnx 失败")

    print("量化 build ...")
    if rknn.build(do_quantization=True, dataset=DATASET) != 0:
        raise RuntimeError("build 失败")

    print("导出 ...")
    if rknn.export_rknn(out_path) != 0:
        raise RuntimeError("export 失败")
    rknn.release()
    print("完成：%s（%.1f MB）" % (out_path, os.path.getsize(out_path) / 1024 / 1024))


if __name__ == "__main__":
    if not os.path.isfile(ONNX):
        print("找不到 ONNX:", ONNX)
        sys.exit(1)
    make_dataset()
    build(os.path.join(PROJ, "models/best_detrac_official.rknn"))
    print("\n转完了。重点看 build 阶段有没有再打 'dtype changed to int8' 的警告。")
