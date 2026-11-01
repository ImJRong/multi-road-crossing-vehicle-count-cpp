#!/usr/bin/env python3
"""转两个 INT8 模型对比：normal vs kl（KL散度）。

关键修正：加了 quantized_input_type / quantized_output_type，
否则 RKNN 会把输入输出压成 int8，C++ 端喂 uint8 会框飞。

用法（WSL 里）：
    source ~/rknn_env.sh
    python ~/traffic_count/tools/convert_fixed.py
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


def build(algo, out_path):
    print("\n" + "=" * 52)
    print("转换：%s -> %s" % (algo, os.path.basename(out_path)))
    print("=" * 52)

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform="rk3588",
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quantized_dtype="w8a8",
        quantized_algorithm=algo,
        quantized_input_type="uint8",       # 关键：输入保持 uint8
        quantized_output_type="float32",    # 关键：输出保持 float32
        optimization_level=1,
        disable_rules=["seperate_conv_rs_split"],
    )

    print("加载 ONNX ...")
    if rknn.load_onnx(model=ONNX) != 0:
        raise RuntimeError("load_onnx 失败")

    print("量化 build（%s）..." % algo)
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
    build("normal", os.path.join(PROJ, "models/best_detrac_fix_normal.rknn"))
    build("kl", os.path.join(PROJ, "models/best_detrac_fix_kl.rknn"))
    print("\n两个模型都转完了，在 models/ 下。")
