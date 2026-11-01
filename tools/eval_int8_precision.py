# 三个模型跑同一批验证图，量 INT8 到底掉多少精度：
#   A  ONNX FP32        models/best_detrac.onnx + onnxruntime
#   B  RKNN FP16        同进程 build(do_quantization=False)
#   C  RKNN INT8        同进程 build(do_quantization=True, dataset=校准数据)
#
# 跑在哪：WSL 的 rknn 环境（Python 3.10，有 rknn-toolkit2 / onnxruntime 1.23.2 / opencv / numpy）
#   source ~/rknn_env.sh
#   cd ~/int8_eval && python eval_int8_precision.py              # 默认：评测 1510 张，校准 200 张
#   cd ~/int8_eval && python eval_int8_precision.py --limit 200  # 先试跑 200 张看速度
#
# 为什么脚本不在项目目录里：rknn-toolkit2 只有 Linux 版，必须放在 WSL 里跑。
# 但数据一张都不用搬 —— 下面的 ROOT 写死 /mnt/d/...，直接跨文件系统去读 D 盘的项目。
#
# 为什么两个 build 在同一个进程：rknn-toolkit2 的模拟器只认当前进程 build 出来的模型，
# 拿 load_rknn 加载现成文件在模拟器上跑不起来。
#
# 为什么要校准数据：INT8 把每层激活值压进 256 个整数格子，格子怎么分要靠一批图统计数值范围。
# 因为 C 的输入沿用 float32 0~1（和 C++ 的 rknn_detector.cpp 保持一致，不改板端代码），
# 校准数据也必须存成 0~1 的 float32，尺度对不上量程就废了。
#
# 输出：
#   models/best_detrac_int8.rknn     转出来的 INT8 模型
#   outputs/calib_200.npy            校准数据（float32 0~1，约 1 GB）
#   outputs/int8_eval_per_image.csv  每张图的框数、配对情况
#   终端打印 mAP50 / mAP50-95 汇总表

import os
import csv
import glob
import time
import argparse

import cv2
import numpy as np

# 项目根目录写死。脚本放在哪个目录都不影响，数据一律去 /mnt/d 读，一张不搬。
# 换机器就改这一行，或者跑之前 export TRAFFIC_ROOT=/新的/路径
ROOT = os.environ.get("TRAFFIC_ROOT", "/mnt/d/multi-road-crossing-vehicle-count")
ONNX_PATH = os.path.join(ROOT, "models", "best_detrac.onnx")
INT8_OUT = os.path.join(ROOT, "models", "best_detrac_int8.rknn")
# 拆分输出头的 ONNX：把末端 Concat 去掉，boxes 和 scores 各当一个输出。
# 为什么需要它：原模型末端 Concat 把 0~640 的坐标和 0~1 的分数拼成一个张量，
# INT8 一个 scale 覆盖 0~640（分辨率 2.5），分数全被压成 0 —— 实测 INT8 分数 max 就是 0.0000。
ONNX_SPLIT = os.path.join(ROOT, "models", "best_detrac_head2.onnx")
VAL_TXT = os.path.join(ROOT, "data", "dataset", "val.txt")
CALIB_DIR = os.path.join(ROOT, "data", "aicity", "images")
OUT_DIR = os.path.join(ROOT, "outputs")

IMGSZ = 640
PAD = 114                      # letterbox 补边灰度，和 C++ / Python 主流程一致
CONF_EVAL = 0.001              # 算 mAP 要完整排序，不能像部署那样卡 0.1
NMS_IOU = 0.7                  # 与 ultralytics 验证时的默认值一致
CONF_PAIR = 0.1                # 配对统计用的阈值，与 configs 里的 conf 对齐
IOU_THRS = np.arange(0.5, 0.96, 0.05)


# ---------------------------------------------------------------- 路径与预处理

def to_wsl(p):
    """D:/xxx -> /mnt/d/xxx。val.txt 里写的是 Windows 路径，WSL 里要换一下才能打开。"""
    p = p.strip().replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def letterbox(img, imgsz=IMGSZ):
    """等比缩放 + 灰边补到 640，返回 (图, scale, dx, dy)。框坐标要靠 scale/dx/dy 还原。"""
    h, w = img.shape[:2]
    s = min(imgsz / w, imgsz / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = cv2.resize(img, (nw, nh))
    dx, dy = (imgsz - nw) // 2, (imgsz - nh) // 2
    out = cv2.copyMakeBorder(canvas, dy, imgsz - nh - dy, dx, imgsz - nw - dx,
                             cv2.BORDER_CONSTANT, value=(PAD, PAD, PAD))
    return out, s, dx, dy


def load_image(path):
    """读成 RGB 并 letterbox。返回 (RGB 图, scale, dx, dy, 原尺寸)。"""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("读不到图: " + path)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lb, s, dx, dy = letterbox(rgb)
    return lb, s, dx, dy, (w, h)


def label_path_of(img_path):
    """val.txt 指到 images/xxx.jpg，标注在 labels/xxx.txt。"""
    d = os.path.dirname(img_path)
    name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    return os.path.join(os.path.dirname(d), "labels", name)


def load_gt(img_path, s, dx, dy, wh, imgsz=IMGSZ):
    """YOLO 标注（归一化 cx cy w h）-> 640 letterbox 空间的 xyxy。没有标注就返回空数组。"""
    lp = label_path_of(img_path)
    if not os.path.exists(lp):
        return np.zeros((0, 4), np.float32)
    W, H = wh
    out = []
    with open(lp, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1 = (cx - bw / 2) * W * s + dx
            y1 = (cy - bh / 2) * H * s + dy
            x2 = (cx + bw / 2) * W * s + dx
            y2 = (cy + bh / 2) * H * s + dy
            out.append([x1, y1, x2, y2])
    return np.array(out, np.float32).reshape(-1, 4)


# ---------------------------------------------------------------- 后处理

def to5(arr):
    """模型输出统一成 (5, 8400)：4 行坐标 + 1 行分数。顺手兼容转置过的布局。"""
    a = np.asarray(arr)
    while a.ndim > 2:
        a = a[0]
    if a.shape[0] != 5 and a.shape[1] == 5:
        a = a.T
    return a.astype(np.float32)


def assemble(outs):
    """把后端返回的输出拼成 (5,8400)。

    单输出模型（原 ONNX / fp16）直接就是 [1,5,8400]。
    双输出模型（拆掉末端 Concat 的）给的是 [1,4,8400] 坐标 + [1,1,8400] 分数，这里拼回去。
    """
    if len(outs) == 1:
        return to5(outs[0])
    a, b = np.asarray(outs[0]), np.asarray(outs[1])
    if a.shape[-2] == 1:          # 顺序反了（分数在前）就换回来
        a, b = b, a
    return np.vstack([a.reshape(4, -1), b.reshape(1, -1)]).astype(np.float32)


def nms_np(boxes, scores, thr):
    """纯 numpy 的 NMS。boxes 是 xyxy。"""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        ar = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        order = rest[inter / (ai + ar - inter + 1e-9) <= thr]
    return np.array(keep, np.int64)


def decode(pred, conf=CONF_EVAL, iou=NMS_IOU):
    """(5,8400) -> (n,5) 的 x1 y1 x2 y2 score，坐标仍在 640 letterbox 空间。"""
    scores = pred[4]
    idx = np.where(scores >= conf)[0]
    if idx.size == 0:
        return np.zeros((0, 5), np.float32)
    p = pred[:, idx].T
    b = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2,
                  p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
    keep = nms_np(b, p[:, 4], iou)
    return np.concatenate([b[keep], p[keep, 4:5]], 1).astype(np.float32)


def iou_xyxy(a, b):
    """a:(m,4) b:(k,4) -> (m,k)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def ap_at(dets_all, gts_all, thr):
    """单类 AP。dets_all/gts_all 都是逐图列表。做法同 VOC：全局按分数排序，逐个判 TP/FP。"""
    npos = sum(len(g) for g in gts_all)
    if npos == 0:
        return 0.0
    recs = []
    for i, d in enumerate(dets_all):
        for j in range(len(d)):
            recs.append((float(d[j, 4]), i, j))
    if not recs:
        return 0.0
    recs.sort(key=lambda x: -x[0])

    used = [np.zeros(len(g), bool) for g in gts_all]
    tp = np.zeros(len(recs), np.float32)
    fp = np.zeros(len(recs), np.float32)
    for k, (_, i, j) in enumerate(recs):
        g = gts_all[i]
        if len(g) == 0:
            fp[k] = 1
            continue
        ious = iou_xyxy(dets_all[i][j:j + 1, :4], g)[0]
        ious[used[i]] = -1.0
        b = int(ious.argmax())
        if ious[b] >= thr:
            tp[k] = 1
            used[i][b] = True
        else:
            fp[k] = 1

    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    rec = ctp / npos
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # 全点插值求曲线下面积
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ---------------------------------------------------------------- 三个推理后端

def run_onnx(items):
    """items: [(图路径, RGB 图, scale, dx, dy, 原尺寸)] -> 逐图 dets"""
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    dets = []
    t0 = time.time()
    for k, (_, img, *_rest) in enumerate(items):
        blob = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))[None]
        dets.append(decode(to5(sess.run(None, {name: blob})[0])))
        progress("ONNX", k, len(items), t0)
    return dets


def make_input(img, mode, fmt):
    """按 (数据类型, 布局) 造一帧输入。mode: float01 / uint8；fmt: nhwc / nchw"""
    a = img.astype(np.float32) / 255.0 if mode == "float01" else img
    if fmt == "nchw":
        a = np.transpose(a, (2, 0, 1))
    return a[None]


def sniff(rknn, img, tag):
    """探这个模型到底认哪种输入。

    为什么要试四种而不是两种：第一次跑 INT8 时，float01 和 uint8 都返回全 0 分数，
    说明除了数据类型，布局（nhwc/nchw）也可能是错的。四种组合各跑一帧，
    谁给出 0~1 之间的分数就用谁，并把四种的结果都打出来，省得来回猜。
    """
    print("  [%s] 四种输入组合各试一帧：" % tag)
    for mode in ("float01", "uint8"):
        for fmt in ("nhwc", "nchw"):
            try:
                outs = rknn.inference(inputs=[make_input(img, mode, fmt)], data_format=fmt)
            except Exception as e:
                print("    %-8s %-5s -> 报错: %s" % (mode, fmt, e))
                continue
            raw = np.asarray(outs[0])
            out = assemble(outs)
            s = out[4]
            print("    %-8s %-5s -> dtype=%s shape=%s" % (mode, fmt, raw.dtype, raw.shape))
            print("                 分数 max=%.6f  mean=%.8f  非零 %.1f%%"
                  % (s.max(), s.mean(), (s > 0).mean() * 100))
            print("                 坐标行 max=%.1f  min=%.1f" % (out[0].max(), out[0].min()))
            if 0.05 < s.max() <= 1.0:
                print("  [%s] 采用：%s + %s" % (tag, mode, fmt))
                return mode, fmt
    raise SystemExit(
        "\n[%s] 四种组合都没有合理输出。按顺序试：\n"
        "  1. 换 uint8 约定（多半是这个）：python %s --calib-format u8\n"
        "     这条会带 mean/std=255 重建模型，输入改成 uint8 0~255；\n"
        "     上板时 rknn_detector.cpp 要改两行，改前先确认。\n"
        "  2. 降优化等级：python %s --calib-format u8 --opt-level 1\n"
        "  3. 都不行就把这段输出发我，别自己瞎改。\n"
        % (tag, os.path.basename(__file__), os.path.basename(__file__)))


def build_rknn(quant, calib_path, tag, opt_level=3, uint8_conv=False, model_path=None):
    from rknn.api import RKNN
    rknn = RKNN(verbose=False)

    cfg = dict(target_platform="rk3588", disable_rules=["seperate_conv_rs_split"],
               optimization_level=opt_level)
    if quant:
        cfg.update(quantized_dtype="asymmetric_quantized-8",
                   quantized_algorithm="normal",
                   quantized_method="channel")
        if uint8_conv:
            # uint8 约定：输入是 0~255 的整数，模型内部除 255。
            # 这条路板端要把 rknn_detector.cpp 改成 RKNN_TENSOR_UINT8，改前先确认。
            cfg.update(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])

    print("\n[%s] config + load_onnx（optimization_level=%d）" % (tag, opt_level))
    if rknn.config(**cfg) != 0:
        raise SystemExit("[%s] config 失败" % tag)
    src = model_path or ONNX_PATH
    print("[%s] 模型: %s" % (tag, src))
    if rknn.load_onnx(model=src) != 0:
        raise SystemExit("[%s] load_onnx 失败" % tag)

    print("[%s] build（%s）... 几分钟，别中断" % (tag, "INT8 量化 + 校准" if quant else "fp16"))
    t0 = time.time()
    ok = rknn.build(do_quantization=True, dataset=calib_path) if quant else rknn.build(do_quantization=False)
    if ok != 0:
        hint = ""
        if quant:
            hint = ("\n\nINT8 build 失败。按顺序试：\n"
                    "  1. 降优化等级再试：python %s --opt-level 1\n"
                    "  2. 换成 jpg 校准：python %s --calib-format jpg（这条会把输入约定变成 uint8 0~255，\n"
                    "     C++ 的 rknn_detector.cpp 必须同步改两行，改之前先确认）\n"
                    % (os.path.basename(__file__), os.path.basename(__file__)))
        raise SystemExit("[%s] build 失败%s" % (tag, hint))
    print("[%s] build 完成 %.0f 秒" % (tag, time.time() - t0))

    if quant:
        if rknn.export_rknn(INT8_OUT) != 0:
            raise SystemExit("[%s] export 失败" % tag)
        print("[%s] 导出 %s (%.2f MB)" % (tag, INT8_OUT, os.path.getsize(INT8_OUT) / 1024 / 1024))

    if rknn.init_runtime() != 0:
        raise SystemExit("[%s] 模拟器起不来" % tag)
    return rknn


def run_rknn(rknn, items, tag):
    mode, fmt = sniff(rknn, items[0][1], tag)
    dets = []
    t0 = time.time()
    for k, (_, img, *_rest) in enumerate(items):
        outs = rknn.inference(inputs=[make_input(img, mode, fmt)], data_format=fmt)
        dets.append(decode(assemble(outs)))
        progress(tag, k, len(items), t0)
    return dets


def progress(tag, k, total, t0):
    """每 20 张报一次，把"还剩多久"算给你看，别干等。"""
    n = k + 1
    if n % 20 and n != total:
        return
    el = time.time() - t0
    per = el / n
    print("  [%s] %d/%d  %.2f 秒/张  已用 %.0f 分  预计还剩 %.0f 分"
          % (tag, n, total, per, el / 60, per * (total - n) / 60))


# ---------------------------------------------------------------- 校准数据

def make_calib(count, fmt):
    files = sorted(glob.glob(os.path.join(CALIB_DIR, "*.jpg")))
    if not files:
        raise SystemExit("校准图目录是空的: " + CALIB_DIR)
    step = max(1, len(files) // count)
    picked = files[::step][:count]
    print("校准图: %d 张（%d 张里按每 %d 张取一张）" % (len(picked), len(files), step))

    if fmt == "jpg":
        d = os.path.join(OUT_DIR, "calib_jpg")
        os.makedirs(d, exist_ok=True)
        paths = []
        for f in picked:
            img = cv2.imread(f)
            lb, _, _, _ = letterbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            out = os.path.join(d, os.path.basename(f))
            # 故意把 RGB 数组直接写盘（不转 BGR）：读回来时数值顺序才是 RGB，
            # 与推理时喂的 RGB 一致。转 BGR 再存的话读回来就是 BGR，颜色是错的。
            cv2.imwrite(out, lb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(out)
        txt = os.path.join(OUT_DIR, "calib_list.txt")
        with open(txt, "w", encoding="utf-8") as fp:
            fp.write("\n".join(paths) + "\n")
        print("校准清单: %s（注意：这条路需要用 mean/std=255，本轮方案默认不走）" % txt)
        return txt

    # npy 路线：每张图存一个 .npy，再给一个 txt 列路径。
    #
    # 为什么不塞成一个大 npy 直接传给 dataset：
    # rknn-toolkit2 2.5.0 的 sparse_weight 那一步会把 dataset 当文本逐行读，
    # 传 .npy 本体就撞上魔数 0x93，报 UnicodeDecodeError: 'utf-8' codec can't decode byte 0x93。
    # dataset 要的是「清单」，数据本身通过清单里的路径去加载。
    #
    # 为什么存 float32 0~1 而不是 0~255 的 jpg：
    # 模型的输入约定是 float 0~1（与 C++ 的 rknn_detector.cpp 一致），
    # 校准数据的尺度必须和推理时一致，否则量化量程整个偏掉。
    # 顺带避开 jpg 路线的 RGB/BGR 争议（quant_img_RGB2BGR 默认 False，图会被当 BGR 用，颜色是错的）。
    d = os.path.join(OUT_DIR, "calib_npy")
    os.makedirs(d, exist_ok=True)
    for old in glob.glob(os.path.join(d, "*.npy")):
        os.remove(old)

    paths = []
    for i, f in enumerate(picked):
        img = cv2.imread(f)
        lb, _, _, _ = letterbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 形状必须是 NCHW (1,3,640,640)，不是 NHWC。
        # 报错过一次：'shape (1,640,640,3) is wrong, expect nchw like (1,3,640,640)'。
        # (2,0,1) 就是把 HWC 换成 CHW 轴序。
        # fmt=npy -> float32 0~1（方案 A，配 float 输入）；fmt=u8 -> uint8 0~255（配 uint8 输入）
        x = np.transpose(lb, (2, 0, 1))[None]
        arr = x.astype(np.float32) / 255.0 if fmt == "npy" else x.astype(np.uint8)
        out = os.path.join(d, "c%04d.npy" % i)
        np.save(out, arr)
        paths.append(out)

    txt = os.path.join(OUT_DIR, "calib_list.txt")
    with open(txt, "w", encoding="utf-8") as fp:
        fp.write("\n".join(paths) + "\n")
    print("校准数据: %d 个 %s npy -> %s（清单 %s）"
          % (len(paths), "float32 0~1" if fmt == "npy" else "uint8 0~255", d, txt))
    return txt


# ---------------------------------------------------------------- 汇总

def pair_stats(a, b):
    """conf>=0.1 的框按 IoU>0.5 贪心配对，返回 (配对数, a 的框数, 平均分数差, 平均中心偏移)。"""
    a = a[a[:, 4] >= CONF_PAIR] if len(a) else a
    b = b[b[:, 4] >= CONF_PAIR] if len(b) else b
    if not len(a) or not len(b):
        return 0, len(a), 0.0, 0.0
    m = iou_xyxy(a[:, :4], b[:, :4])
    order = np.dstack(np.unravel_index(np.argsort(-m, axis=None), m.shape))[0]
    ua, ub, ds, dc = set(), set(), [], []
    for i, j in order:
        if m[i, j] < 0.5:
            break
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        ds.append(b[j, 4] - a[i, 4])
        ca = ((a[i, 0] + a[i, 2]) / 2, (a[i, 1] + a[i, 3]) / 2)
        cb = ((b[j, 0] + b[j, 2]) / 2, (b[j, 1] + b[j, 3]) / 2)
        dc.append(float(np.hypot(ca[0] - cb[0], ca[1] - cb[1])))
    return len(ua), len(a), float(np.mean(ds)), float(np.mean(dc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张验证图，0 = 全部")
    ap.add_argument("--calib", type=int, default=200, help="校准图张数")
    ap.add_argument("--split-head", action="store_true",
                    help="INT8 用拆分输出头的 ONNX（boxes/scores 各一个输出），避免末端 Concat 把分数压成 0")
    ap.add_argument("--opt-level", type=int, default=3, choices=(0, 1, 2, 3),
                    help="RKNN 的 optimization_level，INT8 build 报错时先降这个")
    ap.add_argument("--calib-format", default="npy", choices=("npy", "u8", "jpg"),
                    help="npy = float32 0~1（默认，配方案 A，C++ 不用改）；"
                         "u8 = uint8 0~255 的 npy（配 uint8 输入）；"
                         "jpg = uint8 0~255 的图（最后退路）")
    args = ap.parse_args()

    for p in (ONNX_PATH, VAL_TXT):
        if not os.path.exists(p):
            raise SystemExit("缺文件: " + p)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 验证集清单
    with open(VAL_TXT, encoding="utf-8") as f:
        paths = [to_wsl(l) for l in f if l.strip()]
    if args.limit:
        paths = paths[:args.limit]
    print("验证图: %d 张" % len(paths))

    print("\n=== 读图 + letterbox + 读标注（三个模型共用这份输入）===")
    items, gts = [], []
    t0 = time.time()
    for k, p in enumerate(paths):
        img, s, dx, dy, wh = load_image(p)
        items.append((p, img, s, dx, dy, wh))
        gts.append(load_gt(p, s, dx, dy, wh))
        if (k + 1) % 200 == 0 or k + 1 == len(paths):
            print("  %d/%d  已用 %.0f 秒" % (k + 1, len(paths), time.time() - t0))
    npos = sum(len(g) for g in gts)
    print("标注框合计 %d 个，平均每图 %.2f 个" % (npos, npos / max(len(gts), 1)))

    # 2. 三个模型各跑一遍
    print("\n=== A: ONNX FP32 ===")
    d_onnx = run_onnx(items)

    calib = make_calib(args.calib, args.calib_format)

    print("\n=== B: RKNN FP16 ===")
    rk = build_rknn(False, None, "fp16", args.opt_level)
    d_fp16 = run_rknn(rk, items, "fp16")
    rk.release()

    print("\n=== C: RKNN INT8 ===")
    int8_src = ONNX_SPLIT if args.split_head else ONNX_PATH
    if args.split_head and not os.path.exists(int8_src):
        raise SystemExit("缺 %s\n先把它从 staging 拷进 models/（命令见文档）" % int8_src)
    rk = build_rknn(True, calib, "int8", args.opt_level,
                    uint8_conv=(args.calib_format in ("u8", "jpg")), model_path=int8_src)
    d_int8 = run_rknn(rk, items, "int8")
    rk.release()

    # 3. 算 mAP
    print("\n=== 计算 mAP ===")
    res = {}
    for tag, dets in (("ONNX FP32", d_onnx), ("RKNN FP16", d_fp16), ("RKNN INT8", d_int8)):
        ap50 = ap_at(dets, gts, 0.5)
        aps = [ap_at(dets, gts, t) for t in IOU_THRS]
        res[tag] = (ap50, float(np.mean(aps)))
        print("  %-10s mAP50 %.4f   mAP50-95 %.4f" % (tag, ap50, res[tag][1]))

    # 4. 配对统计 + 逐图明细
    rows, summary = [], {"fp16": [0, 0, [], []], "int8": [0, 0, [], []]}
    for k, p in enumerate(paths):
        pa, na, ds, dc = pair_stats(d_onnx[k], d_fp16[k])
        pb, nb, ds2, dc2 = pair_stats(d_onnx[k], d_int8[k])
        summary["fp16"][0] += pa
        summary["fp16"][1] += na
        summary["int8"][0] += pb
        summary["int8"][1] += nb
        if na:
            summary["fp16"][2].append(ds)
            summary["fp16"][3].append(dc)
        if nb:
            summary["int8"][2].append(ds2)
            summary["int8"][3].append(dc2)
        rows.append(dict(image=os.path.basename(p),
                         onnx=len(d_onnx[k]), fp16=len(d_fp16[k]), int8=len(d_int8[k]),
                         pair_fp16=pa, pair_int8=pb))

    csv_path = os.path.join(OUT_DIR, "int8_eval_per_image.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 5. 打印汇总
    def avg_box(dets):
        return sum(len(d) for d in dets) / max(len(dets), 1)

    def avg_score(dets):
        s = [float(d[:, 4].mean()) for d in dets if len(d)]
        return float(np.mean(s)) if s else 0.0

    print("\n" + "=" * 70)
    print("三模型对比：%d 张验证图，%d 个标注框" % (len(paths), npos))
    print("=" * 70)
    print("%-14s %10s %10s %10s" % ("", "ONNX", "FP16", "INT8"))
    print("%-14s %10.4f %10.4f %10.4f" % ("mAP50", *[res[t][0] for t in ("ONNX FP32", "RKNN FP16", "RKNN INT8")]))
    print("%-14s %10.4f %10.4f %10.4f" % ("mAP50-95", *[res[t][1] for t in ("ONNX FP32", "RKNN FP16", "RKNN INT8")]))
    print("%-14s %10.1f %10.1f %10.1f" % ("每图框数(conf0.001)", avg_box(d_onnx), avg_box(d_fp16), avg_box(d_int8)))
    print("%-14s %10.4f %10.4f %10.4f" % ("平均分数", avg_score(d_onnx), avg_score(d_fp16), avg_score(d_int8)))
    print("\n相对 ONNX 的差异：")
    for tag, key in (("FP16", "fp16"), ("INT8", "int8")):
        m, tot, ds, dc = summary[key]
        print("  %-5s 配对数 %d/%d (%.2f%%)  平均分数差 %+.4f  平均中心偏移 %.2f px"
              % (tag, m, tot, m / max(tot, 1) * 100,
                 float(np.mean(ds)) if ds else 0.0,
                 float(np.mean(dc)) if dc else 0.0))
    print("\n逐图明细: %s" % csv_path)
    print("提醒: 这是 PC 模拟器的软件仿真，不等于板端真机表现；速度数字无参考价值。")


if __name__ == "__main__":
    main()
