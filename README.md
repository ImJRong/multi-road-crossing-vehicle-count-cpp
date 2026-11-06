# traffic_count（C++ 版）

多路口车流量统计的板端实现。读监控视频，检测车辆，跟踪发编号，穿过计数线时按方向计数。

和 Python 版是**两套独立代码**，共用同一份模型和同一套配置文件：

| | Python 版 | C++ 版 |
|---|---|---|
| 推理 | ultralytics | ONNX Runtime（PC）/ RKNN（板子） |
| 跟踪 | ultralytics 内置 ByteTrack | 自实现：卡尔曼预测 + 匈牙利分配 + 两阶段关联（代价 1-IoU，阈值 0.8 / 0.5） |
| 用途 | 调参、验证算法、出报告数据 | 板子上跑 |

## 先看哪一节

本仓库（C++ 版）：https://github.com/ImJRong/multi-road-crossing-vehicle-count-cpp
Python 版：https://github.com/ImJRong/multi-road-crossing-vehicle-count

| 目的 | 去哪 |
|---|---|
| 编译、跑起来 | 本文 §三、§四 |
| 改配置（ROI、计数线、阈值） | 本文 §五 |
| 转模型 / 换模型 | 本文 §六、§七 |
| 批量评测模型、看性能数字 | 本文 §七点五、§七点六 |
| 用网页看实时画面、改参数 | 本文 §十二（单路）、§十二点五（四路分屏） |
| 懂原理（为什么这样设计） | 本文 §八~§十一 |
| 完整踩坑记录、交接文档 | [Python 版 MEMORY.md](https://github.com/ImJRong/multi-road-crossing-vehicle-count/blob/main/MEMORY.md) |
| **框偏移的原理推导（带 11 张图）** | [docs/head4_原理详解.md](https://github.com/ImJRong/multi-road-crossing-vehicle-count/blob/main/docs/head4_%E5%8E%9F%E7%90%86%E8%AF%A6%E8%A7%A3.md) |
| Python 版全部工具脚本 | [tools/README.md](https://github.com/ImJRong/multi-road-crossing-vehicle-count/blob/main/tools/README.md) |

> 跨仓库的链接必须写完整 URL，相对路径只在本仓库内有效。

---

## 一、现在是什么状态（2026-11-06）

| 项 | 状态 |
|---|---|
| RKNN 推理（INT8） | ✅ 18.3 fps |
| **框定位精度** | ✅ **中心偏移 0.03 px（原 6.75 px）** |
| 网页实时显示 | ✅ MJPEG 推流 |
| 跟踪 + 过线计数 | ✅ 能跑 |
| 参数实时调整 | ✅ 网页改 conf/iou 立即生效 |
| 四路并行分屏 | ✅ `traffic_web_quad`，四路各 20 fps（板端实测） |
| 系统状态显示 | ✅ CPU / 温度 / NPU 频率 / 三核负载 |

### 框偏移已解决（2026-11-02）

板端框原先整体偏左上 6~22 px，车越小偏得越多。现已修复，**速度不变**。

**原因**：RKNN 按 `范围/256` 分配量化刻度。原模型输出坐标（0~640），刻度 2.5；
而 DFL 内部的距离只有 0~15，被粗刻度压没了精度。

**解法**：模型输出改为「DFL 距离 0~15」，坐标换算移到 `rknn_detector.cpp` 用 float 完成。
距离成为独立输出后，RKNN 按其真实范围分配刻度 0.047，**精细 53 倍**。

模型权重一个字节未改，量化方式未改，速度未变。

> 完整原理（量化机制推导、11 张图）见 Python 版仓库 `docs/head4_原理详解.md`。

### 计数漏（已收尾）

转弯车曾经常漏计。实测定量结果：改用 head4 后**断号减少 34.3%**
（12135 帧，2559 → 1680 个轨迹编号），计数净增 12，无一条线下降。

即框偏移导致的框抖，是漏计的一个成因，但只解释了约三分之一。
剩余成因未进一步查证，投入产出比不高，暂告一段落。

> **注意**：`count_dir` 只数一个方向**不是缺陷**——这几个路口都是单行道，
> 单方向计数是设计意图。详见 Python 版 `MEMORY.md` 6.1。

### 硬件与环境

| 项 | 值 |
|---|---|
| 板子 | Orange Pi 5 Plus（RK3588，8 核，6 GB） |
| 系统 | Ubuntu 22.04.5，内核 5.10.160-rockchip |
| NPU 运行库 | `/usr/lib/librknnrt.so` **2.5.0** |
| NPU 驱动 | 0.9.6 |
| 板子 IP | **192.168.1.105（静态）** |

---

## 二、目录结构

```
CMakeLists.txt          编译配置（主程序 + rknn_probe）
web.cmake               网页版构建规则（traffic_web_single 与 traffic_web_quad）
src/
  main.cpp              命令行入口：读配置 → 裁 ROI → 循环 → 画图 → 存视频
  web_main_single.cpp   网页版入口（单路）：会话管理（开始/停止/重开）+ 主循环 + 抓拍
  web_main_quad.cpp     网页版入口（四路）：四路并行 + 分屏 + 每路独立参数
  web_server.cpp/.hpp   网页版 HTTP 服务：MJPEG 推流 / 状态 JSON / 控制命令
  config.cpp/.hpp       解析 configs/*.json
  counter.cpp/.hpp      穿线计数：判在哪一侧、死区、换边加一
  tracker.cpp/.hpp      跟踪：卡尔曼预测 + 两阶段关联
  kalman_track.hpp      卡尔曼滤波封装（只有头文件）
  detector.cpp/.hpp     ONNX Runtime 推理（PC）
  rknn_detector.cpp/.hpp RKNN 推理（板子）
  rknn_probe.cpp        模型体检工具（换新模型先跑它，见第六节）
  detection.hpp         检测框结构体，两个推理后端共用
  mini_json.hpp         JSON 解析，不依赖第三方库
  recorder.hpp          过线记录写 csv
web/
  index.html            单路监控台页面（CSS/JS 内嵌，不依赖 CDN）
  quad.html             四路分屏页面（点某一路进该路详情）
```

**为什么拆这么细**：换推理后端时只动一个文件（`detector.cpp` ↔ `rknn_detector.cpp`），`main.cpp` 一行不用改。头文件放声明，源文件放实现，因为每个 `.cpp` 是单独编译的。

---

## 三、编译

依赖 OpenCV、ONNX Runtime（PC）或 librknnrt（板子）。

**PC 上（ONNX Runtime）**

```bash
cmake . && make
```

**板子上（RKNN）**

```bash
cmake . -DUSE_RKNN=ON -DRKNN_ROOT=/usr && make
```

`RKNN_ROOT` 目录下要有 `include/rknn_api.h` 和 `lib/librknnrt.so`。板子上这两个分别在 `src/` 和 `/usr/lib/`，所以编译时还要 `-Isrc`。

⚠️ **换了 `CMakeLists.txt` 里的路径之后，必须删掉 `CMakeCache.txt` 再 cmake**：

```bash
rm -f CMakeCache.txt && cmake . -DUSE_RKNN=ON && make
```

`set(... CACHE ...)` 不会覆盖缓存里的旧值，不删就一直用老路径。

编出来三个可执行文件：

| 文件 | 用途 |
|---|---|
| `traffic_count` | 命令行版，跑完退出 |
| `traffic_web_single` | 网页版，常驻等命令 |
| `rknn_probe` | 模型体检（只在 `USE_RKNN=ON` 时编） |

---

## 四、运行

### 命令行版

```bash
./traffic_count test4     # 跑 test4 配置
./traffic_count           # 不带参数，列出现有配置
```

配置读 `configs/testN.json`，视频读配置里写的路径，模型读 `models/best_detrac.onnx`（RKNN 版读 `.rknn`）。

`configs/*.json` 里的 `output` 三种取值：

| 值 | 行为 |
|---|---|
| `"save"` | 写 mp4 到 `output/` |
| `"none"` | **纯推理测速**，不写不显示（板子上最常用） |
| `"show"` | ⚠️ **板子上一定崩**（没显示器，GTK 初始化失败） |

### 网页版

```bash
./traffic_web_single 8080
```

浏览器打开 `http://<板子IP>:8080`，选配置点「开始」。

**这是板子上唯一能看实时画面的方式**（MJPEG 推流）。

⚠️ 测完必须关掉，不然一直占着 8080：

```bash
pkill -f "[t]raffic_web_single"
```

（注意 `[t]` 这个写法：`pkill -f traffic_web_single` 会把自己那条命令也匹配上，一起杀掉。）

### CPU 调频（影响帧率，重启失效）

默认 `ondemand`，同一段 400 帧实测 **15.6 fps**；设 `performance` 后 **20.6 fps**：

```bash
for p in /sys/devices/system/cpu/cpufreq/policy*; do
    echo performance | sudo tee $p/scaling_governor
done
```

---

## 五、配置在哪改

| 想改什么 | 改哪里 | 改完要做什么 |
|---|---|---|
| 工程根目录、库路径 | `CMakeLists.txt` | 删 `CMakeCache.txt` → `cmake . && make` |
| 视频、ROI、计数线、conf、iou、imgsz | `configs/testN.json` | **不用编译**，直接跑 |
| 跟踪器内部阈值 | `tracker.hpp` 的默认参数 | `make` |

### 参数一览

**在 json 里（不用编译）**

| 参数 | 典型值 | 作用 |
|---|---|---|
| `imgsz` | 640 | 送进模型的边长，必须和模型导出时一致 |
| `conf` | 0.1 ~ 0.4 | 检测的最低分。**降到 0.1 才能让跟踪的第二轮工作** |
| `iou` | 0.3 ~ 0.5 | NMS 阈值。太小会把并排的车合成一辆 |
| `lines[].margin` | 6 ~ 10 | 死区（像素）：离线这么近不判过线，防车压线抖动反复计数 |
| `lines[].count_dir` | `"pos"` / `"neg"` / 不写 | 只数一个方向。不写 = 两个方向都数 |

**在代码里（要重新编译）**

| 参数 | 值 | 位置 |
|---|---|---|
| `high_thresh` | 0.25 | `tracker.hpp`，高分框下限 |
| `low_thresh` | 0.1 | 同上，低分框下限 |
| `new_thresh` | 0.25 | 同上，建新轨迹的门槛 |
| `match_thresh` | 0.8 | 同上，第一轮代价上限（≈ IoU 0.2） |
| `track_buffer` | 30 | 同上，轨迹丢失后活几帧 |
| `kMinTravel` | **35** | 同上，位移小于它算静止目标，既不计数也不画框 |

第二轮匹配的阈值 0.5（≈ IoU 0.5）是硬编码的，和 ultralytics 一致：低分框不可靠，位置必须高度吻合才敢认。

---

## 六、模型转换（转不出来先看这节）

### 唯一验证通过的配方（head4，2026-11-02）

| 项 | 值 |
|---|---|
| 工具 | **rknn-toolkit2 1.6.0+81f21f4d**（不要用 2.5.0，见下文） |
| 输入 ONNX | `best_detrac_head4.onnx`（删掉末端 Concat + DFL 之后的 9 个换算节点） |
| ONNX 输出名 | `/model.22/dfl/Reshape_1_output_0`（**距离** `[1,4,8400]`）+ `/model.22/Sigmoid_output_0`（分数 `[1,1,8400]`） |
| config | `target_platform='rk3588'`、`mean_values=[[0,0,0]]`、`std_values=[[255,255,255]]` |
| build | `do_quantization=True`，其余全默认（w8a8 / normal / optimization_level 默认 1） |
| 校准集 | 300 张，每个视频各 75 张 |
| 产出 | `models/best_detrac_v12_head4.rknn`，**5,969,848 字节** |
| 关键指标 | 距离输出的 `scale = 0.046943`（旧模型是 2.5017） |

**为什么输出距离而不是坐标**：RKNN 按 `范围/256` 分配刻度。
输出坐标（0~640）→ 刻度 2.5；输出距离（0~15）→ 刻度 0.047，**精细 53 倍**。

坐标换算（`anchor ± 距离`，再乘 stride）移到 `rknn_detector.cpp` 用 float 完成。

> 详细推导见 Python 版仓库 `docs/head4_原理详解.md`。

### ★ 换模型不用改代码

`rknn_detector.cpp` 按输出张量名是否含 `"dfl"` 自动识别模型类型：

```cpp
// head4: /model.22/dfl/Reshape_1_output_0  -> 含 "dfl"，走距离换算
// head3: /model.22/Mul_2_output_0          -> 不含，直接用坐标
impl_->boxes_is_dist = std::string(a.name).find("dfl") != std::string::npos;
```

因此 `best_detrac.rknn` 换成 head3 或 head4 都能直接跑，无需重新编译。

### ★ 为什么不能用 Split 拆输出头

原 ONNX 的 `output0` 是 `[1,5,8400]`，末端一个 Concat 把坐标（0~640）和分数（0~1）拼在同一个张量里。INT8 是每个张量一套量化参数，量程被坐标撑到 640 后分数全被舍成 0，所以必须拆。

**拆法有两种，只有一种能用**：

| 拆法 | 图结构 | 结果 |
|---|---|---|
| ❌ 在 Concat **后面**加 Split | `Concat_3 → output0 → Split → boxes/scores` | 1.6.0：scores 的 8400 个 int8 全是 `-128`（= zp = 0.0）；2.5.0：分数正常但框塌 |
| ✅ **删掉** Concat | `Mul_2 → boxes`、`Sigmoid → scores` 直接当输出 | 1.6.0 转出来正常 |

原因：RKNN 编译器会把挂在输出上的 Split 融合掉，两个版本各有各的融错法。删掉 Concat 就没有可融的东西。

脚本自带校验：把两路输出拼回去和原模型 `output0` 逐元素比，最大差值必须是 0。

### 复现

```bash
# Windows 侧：生成 ONNX（每步都带校验）
D:\Python3.12.4\python.exe tools\01_convert\make_head3.py
D:\Python3.12.4\python.exe tools\01_convert\make_head4.py
D:\Python3.12.4\python.exe tools\01_convert\verify_head4_decode.py   # 必须看到差值 0

# WSL 侧：转 RKNN
conda activate rknn16
cd ~/traffic_count
python /mnt/d/multi-road-crossing-vehicle-count/tools/01_convert/convert_head4_probe.py
```

### ★ 为什么不能用 rknn-toolkit2 2.5.0

同一份 ONNX、同一批校准图、同一套 config，只换工具版本：

| 工具 | 结果 |
|---|---|
| 1.6.0 | 框正常 |
| 2.5.0 | **w/h 被放大 13 倍**，通道最大值顶到 637.92（= 255 × 2.5） |

2.5.0 的 `auto_hybrid`（自动混合量化）开关本身正常，但救不了这个缺陷。
详见 Python 版 `MEMORY.md` 5.8 节。

### 试过但没用的（别再重复）

- `quantized_dtype`、`hybrid_level`、`custom_string`、`quantized_input_type` —— 都是猜的，无效
- `mmse` / `kl_divergence` —— 只是换裁剪范围，1 像素级收益；mmse 还容易 OOM
- `optimization_level=3` —— 把 DFL 的归一化融没了，坐标算出天文数字（`cx = -2.18e33`）
- 各种混合量化方案 —— 两个工具版本都不通

---

## 七、模型体检：rknn_probe

换新模型先跑它，别拿业务程序猜。

```bash
cd ~/code
make rknn_probe                                            # 只编它，几秒钟
./rknn_probe models/新模型.rknn data/videos/test1.mp4 200
```

它打印每个输入/输出张量的形状、类型、`scale`、`zp`，然后真跑一帧，把输出**原样摊开**：反量化后的 float + 未换算的 int8。

### 合格标准（第 200 帧，test1）

**head4（当前）**

| 输出 | 合格表现 |
|---|---|
| 距离 float | min ≈0、max ≈11、**均值 ≈2.4** |
| 距离 scale | **0.046943**（这是方案成立的关键指标） |
| scores int8 | 8400 个里约 **60 个非零、9 个 >0.5** |

**head3（对照）**

| 输出 | 合格表现 |
|---|---|
| boxes float | min 0、max ≈630、**均值 ≈182**（塌了会变成 330 左右） |
| boxes scale | 2.501652 |
| scores float | max ≈0.78~0.80 |

**一眼判据**：看 `[OUT1 int8]` 那行的 `==zp(-128)`——**8340 左右 = 好；8400 = 全 0，模型坏了**。

（8400 个候选框里绝大部分本来就是背景，所以"几乎全是 -128"是正常的。）

**判断 4 个通道有没有坏**，用 `tools/02_diagnose/ch_scan.cpp`：

```
正常: ch0 0~630 | ch1 0~615 | ch2 2.5~510 | ch3 2.5~230   （四路各不相同）
坏了: 四路全是 0~638                                        （= 255 × 2.5，被截断）
```

---

## 七点五、批量评测：multi_eval

`rknn_probe` 只看一帧，`multi_eval` 跑完整视频并给出计数统计，用来量化比较两个模型的差别。

**和 `traffic_count` 的区别**：不画框、不存视频、可多路并行，所以快得多。
单路 18~22 fps，四路并行合计 **62~73 fps**。

```bash
cd ~/code
make multi_eval

# 四路并行，各跑前 3600 帧（视频更短的跑完即止）
./multi_eval configs/test1.json configs/test2.json configs/test3.json configs/test4.json \
    --frames 3600 --out output/full_head4

# 换模型对比同一批数据
./multi_eval configs/test1.json configs/test2.json configs/test3.json configs/test4.json \
    --frames 3600 --model models/best_detrac_v7.rknn --out output/full_v7
```

参数：

| 参数 | 作用 |
|---|---|
| `--frames N` | 每路最多跑 N 帧，0 或不写 = 全部 |
| `--model path` | 指定 `.rknn` 模型，默认 `models/best_detrac.rknn` |
| `--out prefix` | 结果写到 `<prefix>_summary.csv` |

实现要点：每个线程创建独立的 `RknnDetector`，按 `i % 3` 绑定到 0/1/2 号 NPU 核
（`RknnDetector::set_core()`），因此四路线程能真正分散到三个核上并行。

结果 CSV 用 `tools/02_diagnose/compare_models.py` 对比，输出断号数和各线计数。

### 2026-11-02 实测结果

完整视频（12135 帧），v7 与 head4 同配置对比：

| 视频 | 帧数 | v7 编号数 | head4 编号数 | 降幅 |
|---|---|---|---|---|
| test1 | 3600 | 553 | **450** | 18.6% |
| test2 | 3600 | 625 | **441** | 29.4% |
| test3 | 2535 | 909 | **539** | 40.7% |
| test4 | 2400 | 472 | **250** | 47.0% |
| **合计** | 12135 | **2559** | **1680** | **34.3%** |

编号数是 `Tracker::total_ids()`，即累计分配过多少轨迹编号。
同一视频同样帧数下，编号越少说明断号越少（一辆车被反复换号）。

计数变化净增 **+12**，无一条线下降。

**结论**：框偏移修复让断号减少约三分之一，这部分漏计随之解决；
剩余漏计另有成因（见第四节「计数漏」）。

---

## 七点六、性能实测：pipeline_bench

单视频流水线：**并行推理 + 串行跟踪**。三段之间用队列连接，帧号作钥匙。

```
读帧线程 ──→ 队列 ──→ K 个推理线程 ──→ 队列 ──→ 跟踪线程
                       (各绑一个 NPU 核)          (单线程串行)
```

```bash
cd ~/code
make pipeline_bench
./pipeline_bench configs/test1.json --workers 10 --frames 900
```

| 参数 | 作用 |
|---|---|
| `--workers N` | 推理线程数（默认 3） |
| `--frames N` | 最多跑多少帧（0 = 全部） |
| `--buf N` | 读帧队列上限（默认 32）。批处理设大，实时流设小 |
| `--realtime N` | **按 N fps 节奏送帧，模拟实时摄像头**（默认 0 = 尽快读） |
| `--no-track` | 只检测不跟踪，用于测跟踪开销 |
| `--model path` | 换模型 |

### 单视频吞吐

| 线程数 | 吞吐 | 相对单线程 |
|---|---|---|
| 1 | 18.4 fps | 1.00× |
| 3 | 65.9 fps | 3.58× |
| 6 | 105.5 fps | 5.73× |
| **10** | **118~121 fps** | **6.6×** |
| 16 | 120.1 fps | 6.53× |

**拐点在 10 线程**，之后不再上升（3 个核已吃满）。

### 延迟要拆开看

| 线程数 | 纯推理 | 排队等待 | 端到端 |
|---|---|---|---|
| 1 | 52.5 ms | 1670 ms | 1725 ms |
| 3 | 49.8 ms | 501 ms | 553 ms |
| 10 | 118.7 ms | 296 ms | 418 ms |

**端到端那 1.7 秒里，1.67 秒是在队列排队，真正算一帧只要 52ms。**
排队时间 ≈ 队列长度 × 单帧耗时（32 × 52ms = 1.66 秒，与实测吻合）。

### 模拟实时摄像头（`--realtime`）

判据看**队列占用率**：送帧速度受节奏限制，吞吐必然约等于送帧速度，
真正能看出「跟不跟得上」的是队列有没有堆起来。

| 送帧速度 | 1 线程 | 3 线程 | 6 线程 | 8 线程 |
|---|---|---|---|---|
| 15 fps | ✅ 65ms | ✅ 62ms | — | — |
| 30 fps | ❌ 队列 100% | ✅ **47ms** | — | — |
| 60 fps | ❌ | ✅ 47ms | — | — |
| 90 fps | ❌ | ❌ | ✅ 62ms | ✅ 68ms |
| 120 fps | ❌ | ❌ | — | ✅ 146ms |

**接摄像头必须开多线程**：单线程只能扛 15fps（单核推理上限 20fps），
**3 线程可扛 60fps，延迟仅 47ms**。

### 跟踪占用多少

| 线程数 | 只检测 | 检测+跟踪 |
|---|---|---|
| 3 | 70.7 fps | 66.7 fps |
| 10 | 117.4 fps | 116.9 fps |

**跟踪几乎不占时间**（约 2~3ms/帧，被 NPU 的时间掩盖），
所以不需要「先全部推理完再批量跟踪」，边跑边追没有额外成本。

### 硬件上限

```
单帧推理（单核）: 42~52 ms  →  单核 20~24 fps
三核理论极限:     约 72 fps（无竞争）
流水线实测峰值:   120 fps（靠队列深度掩盖延迟）
```

**120 fps 已接近这块板子的物理上限。**

### 场景建议

| 场景 | 命令 | 预期 |
|---|---|---|
| 批量跑多个 mp4 | `multi_eval` 四路并行 | 71 fps |
| 单个 mp4 要快 | `pipeline_bench --workers 10 --buf 64` | 118 fps |
| 接 30fps 摄像头 | `--workers 3 --buf 30 --realtime 30` | 47ms 延迟 |
| 接 60fps 摄像头 | `--workers 3 --buf 30 --realtime 60` | 47ms，队列 10% |

### GPU 说明

板子有 **Mali-G610 MP4**（最高 1.0 GHz，`/dev/mali*`、`g18p0` 驱动都在），
但**当前项目没用它**：推理在 NPU，跟踪/计数/画图在 CPU。

唯一值得用的地方是存视频时调硬件编码器（rkmpp）省 CPU。
**不能用来训练** —— Mali 没有可用的 PyTorch 后端，且 6GB 共享内存、25GB/s 带宽不够。

---

## 八、跟踪算法

自己实现的 ByteTrack，一帧的流程：

```
① 卡尔曼给每条轨迹预测位置
② 检测框按分数分两堆：>=0.25 高分，0.1~0.25 低分
③ 第一轮：所有轨迹 × 高分框 → 匈牙利分配
④ 第二轮：第一轮没配上的 Tracked 轨迹 × 低分框 → 匈牙利
⑤ 高分框没人认领的 → 建新轨迹；低分框没人认领的 → 丢弃
⑥ 两轮都没配上的 → lost+1，超过 30 帧删掉
```

几个容易搞错的地方：

- **低分框不能建新轨迹**。树影、反光、广告牌边缘都在低分里，让它们开新户，一帧能冒出十几个假 ID
- **Lost 状态的轨迹仍要参与第一轮**，否则"留着等它回来"是空话
- **匈牙利求出的是完美匹配**，必须再剔除代价超阈值的配对，否则所有轨迹都会配上
- **输出框用卡尔曼滤波后的状态**（`kf.state()`），不要再在外面叠 EMA —— 那是重复平滑，白多 1~2 帧滞后（2026-11-01 删掉的）

和 ultralytics 的差别：ultralytics 用 XYAH（中心 + 宽高比 + 高）作为卡尔曼状态，这里用 XYWH；数学等价，只是参数化不同。

---

## 九、计数逻辑

`counter.cpp` 的 `update()`，一辆车每帧调一次：

1. `near_segment(p, a, b, margin)` —— 点落不落在这条线**这一段**上（把点投影到线段方向，看比例 t 在不在 0~1）
2. `side(p, a, b)` —— 叉积算在哪一侧。正负代表两边，`数值 / 线长` 才是真实像素距离
3. 死区：`|s| / 线长 <= margin` 就跳过这一帧，既不计数也不更新记录
4. 查表 `prev_[(线序号, 车号)]` —— 上次在哪一侧
5. 符号变了 = 换边 = 过线。再看 `count_dir` 决定数不数
6. 不管数没数，都把这次的 `s` 写回表里

`count_dir` 的含义：

- `"pos"` = 只数「负侧 → 正侧」
- `"neg"` = 只数「正侧 → 负侧」
- 不写 = 两个方向都数

`retain(live_ids)` 每帧调一次，把跟踪器已经删掉的车号从 `prev_` 里清掉。不清的话接摄像头连跑几天会攒到几十万条。

---

## 十、实测数据

**PC（Intel 核显，无独显，ONNX Runtime）**

| 视频 | 帧数 | 帧率 | A 线 | B 线 | 备注 |
|---|---|---|---|---|---|
| test3 | 2535 | 18~19 FPS | 243 | 186 | B 线人工真值 ≥190，差 4 次（2%） |
| test2 | 27241 | 17.5 FPS | 842 | 893 | Python 版是 838 / 893，总数差 4 次（0.2%） |

**板子（RK3588，INT8，v7 模型）**

| 条件 | 帧率 | 框/帧 | test1 前 400 帧 |
|---|---|---|---|
| `performance` 调频 | 20.6 fps | 4.1 | A 线 6 |
| `ondemand` 调频 | 15.6 fps | 4.6 | A 线 6 |

单帧 NPU 计算 24.8 ms，三核并行。

---

## 十一、INT8 量化精度（2026-10-22 实测）

**设置**：同一份权重导出三个模型。评测集为验证集 1510 张（10795 个标注框）；
输入为 RGB、letterbox 至 640；评估口径 conf=0.001、NMS IoU=0.7，单类，AP 按全点插值计算 mAP50 与 mAP50-95。

| 模型 | mAP50 | mAP50-95 | 每图框数（conf 0.001） |
|---|---|---|---|
| ONNX FP32 | 0.9426 | 0.7196 | 26.1 |
| RKNN FP16 | 0.9426 | 0.7191 | 26.2 |
| RKNN INT8 | 0.9037 | 0.6125 | 26.2 |

**结果**

- INT8 相对 FP32：mAP50 下降 3.9 个百分点（相对 4.1%），mAP50-95 下降 10.7 个百分点（相对 14.9%）。
- 损失集中于定位精度：坐标量化步长为 2.5 px，匹配框的中心平均偏移 1.86 px；检出能力基本不变，每图框数 26.1 → 26.2。
- FP16 与 FP32 在本口径下无可测差异（mAP50 相同，mAP50-95 差 0.0005），可视为无损。
- 业务侧影响：conf≥0.1 的检出框中 1.9% 无法与 FP32 匹配（12843 中 12596），对应漏计风险；中心偏移折算到 1280 宽画面约 3.7 px，穿线判定基本不受影响。

> 这是 PC 模拟器的数字，板端需上板复测。

---

## 十二、网页监控台（traffic_web_single）

同一套算法，多一个能远程看的入口。

`traffic_count` 是"给个配置就跑到底、跑完退出"，`traffic_web_single` 是"起来之后蹲在那，等网页下命令"。
浏览器里选配置、点开始、拖参数、看画面和过车记录，不用碰命令行。

三个可执行文件共用 `config.cpp` / `counter.cpp` / `tracker.cpp` / `detector.cpp`，
网页版只多一个 `web_server.cpp`（HTTP + MJPEG 推流）。计数、跟踪、推理一行都没复制。

`traffic_web_single` 管一路，`traffic_web_quad` 管四路（见十二点五）。

### 页面上有什么

| 区域 | 功能 |
|---|---|
| 顶栏 | 选配置（自动扫 `configs/*.json`）、开始 / 暂停 / 重新开始 / 停止 |
| 主画面 | MJPEG 实时画面。服务端画好再推，不是前端叠图 |
| 图层 | 检测框 / 跟踪编号 / 计数线与数字 / 检测区域，四项独立开关 |
| 参数 | `conf`、`iou` 两个数字框，改完立即生效 |
| 系统状态 | CPU 占用、温度、NPU/GPU 频率、NPU 三核负载 |
| 计数 | 每条线各自的数字、当前跟踪数、fps、单帧耗时 |
| 过车记录 | 最近 40 条，带抓拍图，点图看大图，Esc 关闭 |

运行中切换配置不用先停止：选另一路点「切换」，旧会话销毁、计数归零、画面直接换掉。
视频播完停在最后一帧，状态转暂停，点「重新开始」重跑一遍。

### 参数改动的边界

**网页上改的只写进内存，`configs/*.json` 一个字节都不动。** 进程重启就回到 json 的值。

| 参数 | 网页能改 | 原因 |
|---|---|---|
| `conf` | 能 | 改变后重建 Detector，约两三百毫秒 |
| `iou` | 能 | 同上 |
| `imgsz` | 不能 | 影响模型输入尺寸，重建代价大 |
| `roi` / `lines` | 不能，只能开关显示 | 改要重建 `Counter`，计数会清零 |
| `draw` 的颜色、线宽 | 不能，只能开关显示 | 属于配置，不属于运行时参数 |

启动一路时先用 json 里的 `detect.conf/iou` 当初始值，保证网页版和 `traffic_count` 跑出来的数字一致；
之后网页改才覆盖它。

### HTTP 接口

全部是 GET，可以直接 curl 测。

| 端点 | 作用 |
|---|---|
| `GET /` | 页面 |
| `GET /stream` | MJPEG 推流 |
| `GET /api/configs` | 列出 `configs/*.json` |
| `GET /api/state` | 各线计数 / 跟踪数 / fps / 当前参数 |
| `GET /api/events` | 最近 200 条过车记录 |
| `GET /snap/<文件名>` | 抓拍小图 |
| `GET /api/start?config=test1` | 开始，或运行中切换到这一路 |
| `GET /api/pause?on=1` | 暂停 / 继续 |
| `GET /api/restart` | 从头跑，计数和事件清零 |
| `GET /api/stop` | 停止，释放视频和模型 |
| `GET /api/param?conf=0.3&iou=0.5` | 改检测参数 |
| `GET /api/overlay?box=1&id=0&line=1&roi=1` | 图层开关 |

### 抓拍图

存 `output/snapshots/`，命名 `帧号_线名_序号.jpg`，128×128，**只画一个检测框**，
不画计数线和编号——记录里只需要看清是哪辆车。

过线记录里的坐标就是框中心，抓拍时按中心最近的那个 track 定位，所以不会抓错车。

### 线程模型

- 一个 accept 线程；每个浏览器客户端一个 detached 线程
- 推流线程靠条件变量等新帧，不空转，也不占着锁写网络
- 主线程跑推理循环；网页的开关、参数通过 `atomic` 变量传进去
- 没有第三方 HTTP 库，`web_server.cpp` 里用 POSIX socket 自己写的
- 客户端提前关页面时忽略 `SIGPIPE`，避免整个进程被打死

---

## 十二点五、四路分屏（traffic_web_quad）

一台板子同时跑四路视频，网页 2×2 分屏看，点某一路进详情页。计数、参数、图层各自独立。

### 编译与启动

```bash
make traffic_web_quad
# 位置参数: [端口] 配置1 配置2 配置3 配置4 [每路fps]
./traffic_web_quad 8081 test1 test2 test3 test4 20
```

不写端口用 8081，不写配置名自动按 test1~test4 补齐，不写 fps 用 20。
后台常驻要用 `setsid`，`nohup ... &` 在非交互 shell 里会被回收：

```bash
setsid ./traffic_web_quad test1 test2 test3 test4 20 > /tmp/quad.log 2>&1 < /dev/null &
```

### 线程分配

每路 2 个推理线程（`kInferPerCam = 2`），四路共 8 个，按 `(路号 × 2 + 槽位) % 3`
轮流绑到 NPU 的三个核上，所以四路能真正并行而不是抢一个核。
另有每路 1 个读帧线程 + 1 个跟踪线程。

### 接口都带 cam 路号

| 端点 | 作用 |
|---|---|
| `GET /` | 分屏页 |
| `GET /stream?cam=N` | 第 N 路 MJPEG（`&thumb=1` 是缩略图） |
| `GET /api/state?cam=N` | 第 N 路状态 |
| `GET /api/param?cam=N&conf=0.3&iou=0.3` | 改第 N 路参数 |
| `GET /api/restart?cam=N` | 第 N 路从头跑 |
| `GET /api/overlay?cam=N&box=1&id=0&line=1&roi=1` | 第 N 路图层开关 |
| `GET /api/start?config=all` | 四路一起开 |
| `GET /api/pause?on=1` | 全部暂停 / 继续 |
| `GET /api/quit` | 退出进程 |

### 参数初值的来路

启动时每路先读自己的 `configs/testN.json`，把 `detect.conf` / `detect.iou` 灌进
`WebServer::init_params()` 当参数面板的初值；之后网页上改只动内存，
**`configs/*.json` 一个字节都不写**，进程重开又回到 json 的值。

单路版用 `params_inited` 标志实现同样的事，四路版直接在配置加载循环里调 `init_params`。
放在视频检查之前，这样某路视频缺失时面板上显示的仍是它的配置值。

### 实测（板端，2026-11-06）

四路各 20 fps 稳定，单帧推理 40~60 ms，进程占 487% CPU（8 核）。

**卡顿的来源**：4 条 MJPEG 常驻推流单条约 500 KB/s，四条合计约 2 MB/s；
浏览器对同一主机只开 6 个并发连接，4 条流占满后状态轮询要排队，页面就越看越卡。
首页本身不慢（13 KB，13~48 ms），状态接口 19~60 ms。
治法是降缩略图帧率/分辨率，或把轮询合并成一次请求——尚未做。

---

## 十三、已知问题

1. **计数漏（已收尾）**：断号减少 34.3%，计数净增 12。剩余成因未查证。详见 `MEMORY.md` 6.1
2. ~~**框轻微向左上偏 1~2 px**~~ —— **已解决**：head4 把中心偏移降到 0.03 px，见第六节
3. **框轻微抖动**：INT8 量化噪声 + 卡尔曼。已删掉多余的 EMA（2026-11-01），只保留卡尔曼一层平滑
4. **端口被占时仍打印「服务已启动」**：`main.cpp` 无条件打印，`web_server.cpp` 其实已经报错了
5. **网页参数只存内存**，不写回 json，重启即恢复
6. **四路分屏页面越看越卡**：4 条 MJPEG 常驻流占满浏览器 6 个并发连接，状态轮询排队。见十二点五

---

## 十四、还没做

- 交叉编译到 aarch64（现在是在板子上直接编）
- ~~多路并行 + 分屏~~ —— **已做**：`traffic_web_quad`，四路各 20 fps，见十二点五
- RTSP 实时流（现在只吃 mp4，见下）
- 事件和计数落库（现在只在内存留 200 条）
- 人工真值全覆盖（目前只有 test3 的 B 线）
- 四路分屏的卡顿（MJPEG 占满浏览器 6 连接，见十二点五）

### 下一步：接实时流

四路分屏已经能跑了，但吃的还是 mp4。真正接摄像头还差下面几步（按顺序做）。

**1. 视频源抽象**（最小改动，收益最大）

`Session` 里 `cv::VideoCapture` 的来源现在写死是配置里的 mp4 路径。改成三种都认：

```json
"video": "rtsp://admin:pass@192.168.1.64:554/Streaming/Channels/101"
"video": "0"                                  // 本地 USB 摄像头
"video": "data/videos/test1.mp4"              // 保持现状
```

`cv::VideoCapture` 本身三者通吃，判断一下字符串是不是纯数字、是不是带 `://` 就行。
真正的差别在**读不到帧怎么办**：mp4 读完了就结束，RTSP 断流要重连，不能把会话销毁。

**2. 多会话管理**（已完成）

`web_main_quad.cpp` 里是 `std::array<Cam, kMaxCams>`，每路读帧/推理/跟踪各一个线程，
`WebServer` 只有一个、多路共用一个端口。`Detector` 每路一份，RKNN 上下文一路一个。

**3. 接口加路号**（已完成）

实际用的是 `cam=` 而不是 `ch=`，见十二点五的接口表。

**4. 带宽要重新算**（真的撞上了）

缩略图宽 480 px 时单条约 500 KB/s，四条合计约 2 MB/s，四条流就把浏览器的
6 个并发连接占满了，状态轮询只能排队。降到每路 640 宽 / 5 fps 约 150 KB/s，
或换架构（前端拉子码流，分析服务只推坐标 JSON 叠框）。

**5. 前端**（已完成）

`web/quad.html`：2×2 网格，每格一条 `/stream?cam=N&thumb=1`；点某格进单路详情，
就是单路版那个页面加一个"返回"。仍待改进的是带宽——见十二点五「卡顿的来源」。
