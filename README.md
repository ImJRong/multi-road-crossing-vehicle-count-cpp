# traffic_count（C++ 版）

多路口车流量统计的板端实现。读监控视频，检测车辆，跟踪发编号，穿过计数线时按方向计数。

和 Python 版是**两套独立代码**，共用同一份模型和同一套配置文件：

| | Python 版 | C++ 版 |
|---|---|---|
| 推理 | ultralytics | ONNX Runtime（PC）/ RKNN（板子） |
| 跟踪 | ultralytics 内置 ByteTrack | 自己实现的 ByteTrack |
| 用途 | 调参、验证算法、出报告数据 | 板子上跑 |

## 目录结构

```
CMakeLists.txt          编译配置
src/
  main.cpp              入口：读配置 → 裁 ROI → 循环 → 画图 → 存视频
  config.cpp/.hpp       解析 configs/*.json
  counter.cpp/.hpp      穿线计数：判在哪一侧、死区、换边加一
  tracker.cpp/.hpp      跟踪：卡尔曼预测 + 两阶段关联
  detector.cpp/.hpp     ONNX Runtime 推理（PC）
  rknn_detector.cpp/.hpp RKNN 推理（板子）
  detection.hpp         检测框结构体，两个推理后端共用
  kalman_track.hpp      卡尔曼滤波封装（只有头文件）
  mini_json.hpp         JSON 解析，不依赖第三方库
  recorder.hpp          过线记录写 csv
```

**为什么拆这么细**：换推理后端时只动一个文件（`detector.cpp` ↔ `rknn_detector.cpp`），`main.cpp` 一行不用改。头文件放声明，源文件放实现，因为每个 `.cpp` 是单独编译的。

## 编译

依赖 OpenCV、ONNX Runtime（PC）或 librknnrt（板子）。

**PC 上（ONNX Runtime）**

```bash
cmake . && make
```

**板子上（RKNN）**

```bash
cmake . -DUSE_RKNN=ON -DRKNN_ROOT=/路径/到/rknn_runtime && make
```

`RKNN_ROOT` 目录下要有 `include/rknn_api.h` 和 `lib/librknnrt.so`。

换过开关之后必须先 `cmake` 再 `make`，直接 make 不会重读 CMakeLists。

## 运行

```bash
./traffic_count test4     # 跑 test4 配置
./traffic_count           # 不带参数，列出现有配置
```

配置读 `configs/testN.json`，视频读配置里写的路径，模型读 `models/best_detrac.onnx`（RKNN 版读 `.rknn`）。

**这两个路径都是数据资产，不在本仓库里**，跟 Python 版共用同一份。

## 配置在哪改

| 想改什么 | 改哪里 | 改完要做什么 |
|---|---|---|
| 工程根目录、库路径 | `CMakeLists.txt` | `cmake . && make` |
| 模型文件 | `main.cpp` 的 `TRAFFIC_MODEL` | `make` |
| 视频、ROI、计数线、conf、iou、imgsz | `configs/testN.json` | **不用编译**，直接跑 |

## 参数一览

**在 json 里（不用编译）**

| 参数 | 典型值 | 作用 |
|---|---|---|
| `imgsz` | 640 | 送进模型的边长，必须和模型导出时一致 |
| `conf` | 0.1 | 检测的最低分。**降到 0.1 才能让跟踪的第二轮工作** |
| `iou` | 0.5 | NMS 阈值。太小会把并排的车合成一辆 |

**在代码里（要重新编译）**

| 参数 | 值 | 位置 |
|---|---|---|
| `high_thresh` | 0.25 | `tracker.hpp`，高分框下限 |
| `low_thresh` | 0.1 | 同上，低分框下限 |
| `new_thresh` | 0.25 | 同上，建新轨迹的门槛 |
| `match_thresh` | 0.8 | 同上，第一轮代价上限（≈ IoU 0.2） |
| `track_buffer` | 30 | 同上，轨迹丢失后活几帧 |
| `kMinTravel` | 4.0 | 同上，位移小于它算静止目标，不计数 |

第二轮匹配的阈值 0.5（≈ IoU 0.5）是硬编码的，和 ultralytics 一致：低分框不可靠，位置必须高度吻合才敢认。

## 跟踪算法

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

和 ultralytics 的差别：ultralytics 用 XYAH（中心 + 宽高比 + 高）作为卡尔曼状态，这里用 XYWH；数学等价，只是参数化不同。

## 实测数据

CPU（Intel 核显，无独显）：

| 视频 | 帧数 | 帧率 | A 线 | B 线 | 备注 |
|---|---|---|---|---|---|
| test3 | 2535 | 18~19 FPS | 243 | 186 | B 线人工真值 ≥190 |
| test2 | 27241 | 约 17 FPS | — | — | 待测 |

**已知差距**：远处小目标、被遮挡的车检不出来。test3 的 B 线少 4 次（2%），原因是检测能力，不是跟踪——改参数、换跟踪算法都没有改善，见下。

**被排除的假设**（都有实测数据）：

| 假设 | 实验 | 结果 |
|---|---|---|
| NMS 太狠，并排车被合并 | iou 0.25 → 0.5 | 平均框数 16.8 一点没变，NMS 从未删过相邻车 |
| 静止过滤门槛太高 | kMinTravel 12 → 4 | 计数 -2，噪声级别 |
| 匹配门槛太高导致断号 | iou_thr 0.3 → 0.2 | +2 |
| 贪心匹配串号 | 加 100 像素距离门限 | 0 变化 |
| 记忆太短 | max_lost 15 → 40 | **-64**，僵尸轨迹抢框，明显变差 |

## RKNN 状态

- 模型已转换：`models/best_detrac.rknn`，fp16 不量化，8.5 MB
- 转换脚本在同仓库的 Python 版里：`tools/build_rknn.py`，参数写死可复现
- 模拟器验证：输入 **float32 RGB，取值 0~1**，输出与原始 ONNX **完全一致**（同一帧 143 个框，坐标对到小数点后一位）
- **C++ 的 RKNN 代码已写，未编译验证，未上板**

输入格式搞错的后果：喂 uint8（0~255）会得到 835 个高分框（正常是 143 个），模型认为到处是车；喂 BGR 会少 7% 的框。

## 还没做

- 交叉编译到 aarch64（需要板子上的 OpenCV 和 librknnrt.so）
- 板端实测帧率、功耗、多路并发
- 人工真值全覆盖（目前只有 test3 的 B 线）
