#!/usr/bin/env python3
"""用「删掉末端 Concat」的 ONNX 重转 INT8 模型（复现旧模型的转法）。

和 convert_v6.py 唯一的区别：输入换成 best_detrac_head3.onnx
  head2（现在失败的）:  Concat_3 -> output0 -> Split -> boxes/scores
  head3（这份脚本用的）: 删掉 Concat_3，两个张量直接当输出

参数完全照旧模型（从 best_detrac.rknn 的 custom_string 反推）：
  target_platform=rk3588, mean=0, std=255, 1.6.0 默认 w8a8

用法（WSL 里）：
    conda activate rknn16
    python /mnt/d/multi-road-crossing-vehicle-count/tools/convert_head3.py
产出：~/traffic_count/models/best_detrac_v7.rknn
日志：~/traffic_count/v7_build.log（build 过程同时写到屏幕和日志）
"""
import os
import sys

from rknn.api import RKNN

ONNX = "/mnt/d/multi-road-crossing-vehicle-count/models/best_detrac_head3.onnx"
CALIB_DIR = "/mnt/d/rknn_calib"
DATASET = os.path.expanduser("~/traffic_count/dataset_calib_head3.txt")
OUT = os.path.expanduser("~/traffic_count/models/best_detrac_v7.rknn")
LOG = os.path.expanduser("~/traffic_count/v7_build.log")


class Tee:
    """屏幕 + 日志各写一份，出问题能回看。"""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def make_dataset():
    imgs = sorted(f for f in os.listdir(CALIB_DIR) if f.lower().endswith(".jpg"))
    with open(DATASET, "w", encoding="utf-8") as f:
        for name in imgs:
            f.write(os.path.join(CALIB_DIR, name) + "\n")
    print("校准图 %d 张 -> %s" % (len(imgs), DATASET))


def main():
    if not os.path.isfile(ONNX):
        print("找不到 ONNX:", ONNX)
        return 1

    make_dataset()
    tee = Tee(LOG)
    sys.stdout = tee
    try:
        rknn = RKNN(verbose=False)
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform="rk3588",
        )
        print("加载 ONNX:", ONNX)
        if rknn.load_onnx(model=ONNX) != 0:
            raise RuntimeError("load_onnx 失败")
        print("量化 build ...")
        if rknn.build(do_quantization=True, dataset=DATASET) != 0:
            raise RuntimeError("build 失败")
        print("导出 ->", OUT)
        if rknn.export_rknn(OUT) != 0:
            raise RuntimeError("export 失败")
        rknn.release()
        print("完成：%s（%d 字节）" % (OUT, os.path.getsize(OUT)))
        print("\n下一步（Windows 这边不用你做，我拿到模型会自己验）：")
        print("  cd ~/code && /tmp/rknn_probe models/best_detrac_v7.rknn data/videos/test1.mp4 200")
    finally:
        sys.stdout = tee.stdout
        tee.flush()
        tee.f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
