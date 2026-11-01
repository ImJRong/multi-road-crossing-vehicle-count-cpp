#!/usr/bin/env python3
"""输出层保持 fp16，避免 int8 输出反量化出错。

用法（WSL 里）：
    source ~/rknn_env.sh
    python ~/traffic_count/tools/convert_outfp16.py
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


def build(algo, custom, out_path, extra=None):
    print("\n" + "=" * 54)
    print("转换：algo=%s -> %s" % (algo, os.path.basename(out_path)))
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
    if custom:
        kwargs["custom_string"] = custom
    if extra:
        kwargs.update(extra)
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

    # 方案A：输出两个节点保持 fp16
    build("normal", ["boxes:fp16", "scores:fp16"],
          os.path.join(PROJ, "models/best_detrac_outfp16.rknn"))

    # 方案B：混合量化，让敏感层（含输出附近）自动保持 fp16
    build("normal", None,
          os.path.join(PROJ, "models/best_detrac_hybrid.rknn"),
          extra=dict(quantized_hybrid_level=1, float_dtype="float16"))

    print("\n转完了，两个模型都在 models/ 下。")
