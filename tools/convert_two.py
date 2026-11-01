#!/usr/bin/env python3
"""转两个 INT8 模型做对比：normal 算法 vs mmse 算法。

用法（在 WSL 里）：
    source ~/rknn_env.sh
    python ~/traffic_count/tools/convert_two.py
"""
import os
import sys

from rknn.api import RKNN

PROJ = os.path.expanduser("~/traffic_count")
ONNX = os.path.join(PROJ, "models/best_detrac_head2.onnx")
CALIB_DIR = "/mnt/d/rknn_calib"
DATASET = os.path.join(PROJ, "dataset_calib.txt")


def make_dataset():
    if not os.path.isdir(CALIB_DIR):
        print("找不到校准图目录:", CALIB_DIR)
        sys.exit(1)
    imgs = sorted(f for f in os.listdir(CALIB_DIR) if f.lower().endswith(".jpg"))
    with open(DATASET, "w", encoding="utf-8") as f:
        for name in imgs:
            f.write(os.path.join(CALIB_DIR, name) + "\n")
    print("校准图 %d 张 -> %s" % (len(imgs), DATASET))
    return len(imgs)


def build(algo, out_path):
    print("\n" + "=" * 50)
    print("开始转换：%s 算法 -> %s" % (algo, os.path.basename(out_path)))
    print("=" * 50)

    rknn = RKNN(verbose=False)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quantized_dtype="w8a8",
        quantized_algorithm=algo,
        optimization_level=1,
        disable_rules=["seperate_conv_rs_split"],
        target_platform="rk3588",
    )

    print("加载 ONNX ...")
    if rknn.load_onnx(model=ONNX) != 0:
        raise RuntimeError("load_onnx 失败")

    print("量化 build（%s，可能要几分钟）..." % algo)
    if rknn.build(do_quantization=True, dataset=DATASET) != 0:
        raise RuntimeError("build 失败")

    print("导出 ...")
    if rknn.export_rknn(out_path) != 0:
        raise RuntimeError("export 失败")

    rknn.release()
    size = os.path.getsize(out_path) / 1024 / 1024
    print("完成：%s（%.1f MB）" % (out_path, size))


if __name__ == "__main__":
    if not os.path.isfile(ONNX):
        print("找不到 ONNX:", ONNX)
        sys.exit(1)
    make_dataset()
    build("normal", os.path.join(PROJ, "models/best_detrac_int8_normal.rknn"))
    build("mmse", os.path.join(PROJ, "models/best_detrac_int8_mmse.rknn"))
    print("\n两个模型都转完了，看 models/ 目录。")
