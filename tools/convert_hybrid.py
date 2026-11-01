#!/usr/bin/env python3
"""用混合量化（quantized_hybrid_level）转模型，让输入保持 uint8。

RKNN 2.5.0 的 config 没有 quantized_input_type 参数（那是老版本/其他平台的），
控制输入 dtype 的是量化粒度本身：全 int8 会把输入也压成 int8，
加上 hybrid_level 让部分层（含输入层）保持 fp16，输入就还是 uint8。

用法（WSL 里）：
    source ~/rknn_env.sh
    python ~/traffic_count/tools/convert_hybrid.py
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


def build(algo, hybrid, out_path):
    print("\n" + "=" * 54)
    print("转换：algo=%s hybrid=%d -> %s" % (algo, hybrid, os.path.basename(out_path)))
    print("=" * 54)

    rknn = RKNN(verbose=False)
    kwargs = dict(
        target_platform="rk3588",
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quantized_dtype="w8a8",
        quantized_algorithm=algo,
        optimization_level=1,
        disable_rules=["seperate_conv_rs_split"],
    )
    if hybrid > 0:
        kwargs["quantized_hybrid_level"] = hybrid
        kwargs["float_dtype"] = "float16"
    rknn.config(**kwargs)

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
    build("normal", 1, os.path.join(PROJ, "models/best_detrac_hy1.rknn"))
    build("kl", 1, os.path.join(PROJ, "models/best_detrac_hy1_kl.rknn"))
    print("\n转完了，看 models/ 下 best_detrac_hy1*.rknn")
