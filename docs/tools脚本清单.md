# tools 目录说明

> 2026-11-02 重新整理。按用途分成 5 个子目录。
> **只想转模型的话，看 `01_convert/` 就够了。**

```
tools/
├── 01_convert/     转换主线：生成 ONNX、转 RKNN
├── 02_diagnose/    诊断验证：判断模型好坏、量化误差
├── 03_failed/      失败记录：试过的死路，留着备查，别再跑
├── 04_dataset/     数据集相关：预处理、切视频、抽帧
└── 05_figures/     画图：生成文档里的讲解图
```

---

## 01_convert —— 转换主线（能用）

**按顺序跑这 4 个，就能从原始 ONNX 得到可用的板端模型。**

| 脚本 | 作用 | 跑在哪 |
|---|---|---|
| `make_head3.py` | 从 `best_detrac.onnx` 删掉末端 Concat → `head3.onnx` | Windows |
| `make_head4.py` | 从 head3 再删掉 DFL 之后的 9 个坐标换算节点 → `head4.onnx` | Windows |
| `verify_head4_decode.py` | 验证「距离→坐标」换算公式，**差值必须是 0** | Windows |
| `convert_head4_probe.py` | 用 rknn-toolkit2 **1.6.0** 转 RKNN | WSL |

### 完整流程

```bash
# ===== Windows 侧 =====
cd D:\multi-road-crossing-vehicle-count
D:\Python3.12.4\python.exe tools\01_convert\make_head3.py
D:\Python3.12.4\python.exe tools\01_convert\make_head4.py
D:\Python3.12.4\python.exe tools\01_convert\verify_head4_decode.py
#   ↑ 必须看到「最大差值 = 0」才算过

# ===== WSL 侧 =====
conda activate rknn16
cd ~/traffic_count
python /mnt/d/multi-road-crossing-vehicle-count/tools/01_convert/convert_head4_probe.py
#   产出：~/traffic_count/models/best_detrac.rknn
```

### head4 是什么

把模型输出从「坐标 0~640」换成「DFL 距离 0~15」。

因为 RKNN 配量化刻度是 `范围 / 256`：
- 输出坐标（0~640）→ 刻度 2.5 → 精度被压扁
- 输出距离（0~15）→ 刻度 0.047 → **细 53 倍**

原理详见 `docs/head4_原理详解.md`。

---

## 02_diagnose —— 诊断验证

**换新模型先跑这些，别拿业务程序猜。**

### 板子上跑（要先传到 `/tmp` 再编译）

| 工具 | 用途 |
|---|---|
| `ch_scan.cpp` | **判断模型是否坏掉最快的方法**：打印 boxes 四个通道各自的范围 |
| `box3.cpp` | 两个模型 + FP32 真值三方对比，逐框算误差 |
| `box_cmp.cpp` | 两个模型互相比对（不需要真值） |

编译方法（板子重启后 `/tmp` 会清空，要重传）：

```bash
g++ -O2 -std=c++17 -I~/code/src /tmp/ch_scan.cpp -o /tmp/ch_scan \
    -lrknnrt $(pkg-config --cflags --libs opencv4)
```

### PC 上跑

| 脚本 | 用途 |
|---|---|
| `dump_truth_boxes.py` | 用 FP32 ONNX 生成真值框文件 |
| `batch_box_eval.py` | 批量跑多帧，汇总 v7 与 head4 的平均误差（会调上面的 dump） |
| `compare_models.py` | **对比两个模型的评测结果**（断号数 + 各线计数），读 multi_eval 的 CSV |
| `trace_layers.py` | 逐层 dump 量化误差（`accuracy_analysis`），查哪层崩了 |
| `cmp_trace.py` | 对比两个 toolkit 版本的逐层误差 |
| `show_cfg_scale.py` | 从 `quantization.cfg` 里精确提取指定层的 min/max/scale |
| `verify_norm.py` | 校验 mean/std 归一化是否生效 |

### 怎么判断模型好坏

```bash
cd ~/code
./rknn_probe models/模型.rknn data/videos/test1.mp4 200
```

| 看什么 | 好的样子 | 坏的样子 |
|---|---|---|
| `[OUT1 int8]` 的 `==zp(-128)` | ≈ 8340 | 8400（分数全 0） |
| `[OUT0]` 的 `scale` | head4: 0.047 | 2.5（刻度没变细） |
| `[OUT0 float]` 的 `mean` | head3: ~182 / head4: ~2.4 | 330、380（框被撑大） |

**更准的用 `ch_scan`**，看 4 个通道范围是否各不相同：

```
正常: ch0 0~630 | ch1 0~615 | ch2 2.5~510 | ch3 2.5~230
坏了: 四路全是 0~638（= 255 × 2.5，int8 饱和值）
```

---

## 03_failed —— 失败记录（**别再跑**）

这里的脚本**产出的模型都不能用**，留着是因为：

1. 记录了完整踩坑过程，避免下次重来
2. 万一以后 rknn-toolkit2 修了 bug，这套流程可以直接复用

| 脚本 | 状态 |
|---|---|
| `convert_v10_v11.py` | 2.5.0 转 INT8 和 auto_hybrid 对照。**能跑通，但两个模型都是坏的** |
| `_smoke_auto_hybrid*.py` | 验证 `auto_hybrid` 开关真的会切 fp16 层（小模型冒烟测试） |
| `_verify_params.py` | 打印已安装 rknn 包的真实 API 签名，不靠记忆 |
| `hybrid_250.py` / `hybrid_250_step2.py` | 2.5.0 的手动混合量化，**能生成模型但输出是坏的** |
| `fix_cfg_yaml.py` | 修 cfg 的 YAML 结构 + 标记 DFL 为 fp16 |
| `convert_head3_hybrid.py` / `hybrid_step2.py` | 1.6.0 版，step2 必定崩 |
| `fix_hybrid_cfg.py` / `patch_dtype_fp16.py` | 早期尝试，已废弃 |
| `build_rknn*.py` / `convert_v5.sh` / `convert_head3.py` | 10-30 ~ 11-01 的历史尝试 |

### 两个重要的结论

**① 2.5.0 自身有缺陷，不是混合量化的问题**

同一份 ONNX、同一批校准图、同一套 config，只换工具版本：
- 1.6.0 → 框正常
- 2.5.0 → w/h 被撑大 13 倍

`convert_v10_v11.py` 转出来的模型，输出和 10-31 那个 `int8_normal.rknn` **逐字节相同**。
说明这是稳定复现的 bug，不是偶发。

**② `auto_hybrid` 开关本身是好的**

冒烟测试证明它能正确切层：阈值调严后模型从 38518 → 34153 字节。
它救不了 2.5.0 自身的缺陷，但开关本身没毛病。

失败原因详见 `MEMORY.md` 5.8 节。

---

## 04_dataset —— 数据集相关

| 脚本 | 用途 |
|---|---|
| `prepare_detrac.py` | 准备 Detrac 数据集 |
| `split_dataset.py` | 划分训练/验证集 |
| `prelabel.py` | 预标注 |
| `extract_frames.py` | 从视频抽帧 |
| `cut_video.py` | 切视频片段 |
| `speed_video.py` | 调整视频速度 |
| `read_roi.py` / `read_line.py` | 交互式读 ROI / 计数线坐标 |
| `make_line_base.py` / `sweep_line.py` / `sweep_slope.py` | 调计数线用 |
| `grab_frame.py` | 抓单帧 |
| `check_gpu.py` / `check_obstacle.py` | 环境检查 |
| `compare_cpp_python.py` | C++ 和 Python 版结果对拍 |
| `kaggle_*.py` / `kaggle_train.ipynb` | Kaggle 训练相关 |

---

## 05_figures —— 画讲解图

| 脚本 | 产出 |
|---|---|
| `draw_explain_figs.py` | `docs/figures/fig1~fig8`（8 张原理图） |
| `draw_causal_figs.py` | `docs/figures/fig9~fig11`（因果链、三种输出、为什么能改） |

```bash
D:\Python3.12.4\python.exe tools\05_figures\draw_explain_figs.py
D:\Python3.12.4\python.exe tools\05_figures\draw_causal_figs.py
```

需要 matplotlib + 中文字体（用 Microsoft YaHei 或 SimHei）。

---

## 常见问题

**Q: 脚本报 `FileNotFoundError: ...\tools\models\best_detrac_head3.onnx`**

项目根目录算错了。脚本用 `dirname` 往上退三层算 ROOT，改动目录结构后要检查。

**Q: 板子上 `rknn_probe` 的 `head` 只打印 12 个值，看不出哪个通道坏了**

用 `ch_scan.cpp`，它按通道独立统计 min/max。

**Q: 2.5.0 转模型报 `IndexError: list index out of range`**

`_p_seperate_conv_rs_split` 规则的 bug，加这个参数绕开：

```python
rknn.config(..., disable_rules=["seperate_conv_rs_split"])
```

注意 1.6.0 **不认**这个参数，会报 `not supported`。

---

## C++ 侧的性能测试工具（不在本仓库）

这两个是板端 C++ 程序，源码在 **C++ 仓库** `src/` 下：

| 工具 | 用途 |
|---|---|
| `multi_eval.cpp` | 多路并行评测。每路一线程，按 `i % 3` 绑 NPU 核。不画图不存视频，四路并行 71 fps |
| `pipeline_bench.cpp` | 单视频流水线。并行推理 + 串行跟踪，支持 `--realtime N` 模拟实时摄像头 |

用法和完整性能数据见 C++ 仓库 README 的 **§七点五、§七点六**，
本仓库 `MEMORY.md` 的 **六点五节** 也有同样一份数据。
