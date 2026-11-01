#!/usr/bin/env python3
"""转 w8a16 模型（激活用 fp16，输入不会被压成 int8，代码不用改）。

用法（WSL 里）：
    source ~/rknn_env.sh
    python ~/traffic_count/tools/convert_w8a16.py
"""
import os
import sys

from rknn.api import RKNN

PROJ = os.path.expanduser("~/traffic_count")
ONNX = os.path.join(PROJ, "models/best_detrac_head2.onnx")
CALIB_DIR = "/mnt/d/rknn_calib"
DATASET = os.path.join(PROJ, "dataset_calib.txt")
OUT = os.path.join(PROJ, "models/best_detrac_w8a16.rknn")


def make_dataset():
    imgs = sorted(f for f in os.listdir(CALIB_DIR) if f.lower().endswith(".jpg"))
    with open(DATASET, "w", encoding="utf-8") as f:
        for name in imgs:
            f.write(os.path.join(CALIB_DIR, name) + "\n")
    print("校准图 %d 张" % len(imgs))


def main():
    make_dataset()

    rknn = RKNN(verbose=False)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quantized_dtype="w8a16",          # 关键：激活保留 16 位，输入仍是 uint8/float
        quantized_algorithm="normal",
        optimization_level=1,
        disable_rules=["seperate_conv_rs_split"],
        target_platform="rk3588",
    )

    print("加载 ONNX ...")
    if rknn.load_onnx(model=ONNX) != 0:
        raise RuntimeError("load_onnx 失败")

    print("量化 build（w8a16，可能 5~10 分钟）...")
    if rknn.build(do_quantization=True, dataset=DATASET) != 0:
        raise RuntimeError("build 失败")

    print("导出 ...")
    if rknn.export_rknn(OUT) != 0:
        raise RuntimeError("export 失败")
    rknn.release()

    print("完成：%s（%.1f MB）" % (OUT, os.path.getsize(OUT) / 1024 / 1024))


if __name__ == "__main__":
    main()
