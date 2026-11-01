# traffic_count（C++ 版）

多路口车流量统计的板端实现。读监控视频，检测车辆，跟踪发编号，穿过计数线时按方向计数。

和 Python 版是**两套独立代码**，共用同一份模型和同一套配置文件：

| | Python 版 | C++ 版 |
|---|---|---|
| 推理 | ultralytics | ONNX Runtime（PC）/ RKNN（板子） |
| 跟踪 | ultralytics 内置 ByteTrack | 自实现：卡尔曼预测 + 匈牙利分配 + 两阶段关联（代价 1-IoU，阈值 0.8 / 0.5） |
| 用途 | 调参、验证算法、出报告数据 | 板子上跑 |

> 更细的部署记录（踩过的坑、每天改了什么、现在卡在哪）在 Python 版的 `MEMORY.md` 里。

---

## 一、现在是什么状态（2026-11-01）

### 板子上跑通了

| 项 | 状态 |
|---|---|
| RKNN 推理（INT8） | ✅ 跑通，`output: none` 时 20.6 fps |
| 网页实时显示 | ✅ MJPEG 推流，浏览器能看 |
| 跟踪 + 过线计数 | ✅ 能跑 |
| 参数实时调整 | ✅ 网页改 conf/iou 立即生效 |
| 系统状态显示 | ✅ CPU / 温度 / NPU 频率 / 三核负载 |

### 主要问题

**计数漏**：转弯车经常漏计，直行车正常。还没解决，怀疑在 `counter.cpp` 的 `near_segment`（车从线段端点外侧绕过去就判不到）或 `count_dir`（只数一个方向）。详见 Python 版 `MEMORY.md` 第六节。

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
web.cmake               网页版 traffic_web_single 的构建规则
src/
  main.cpp              命令行入口：读配置 → 裁 ROI → 循环 → 画图 → 存视频
  web_main_single.cpp   网页版入口：会话管理（开始/停止/重开）+ 主循环 + 抓拍
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
  index.html            网页监控台页面（CSS/JS 内嵌，不依赖 CDN）
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

### 唯一验证通过的配方

| 项 | 值 |
|---|---|
| 工具 | **rknn-toolkit2 1.6.0+81f21f4d**（不是 2.5.0） |
| 输入 ONNX | `best_detrac_head3.onnx`（**删掉了末端 Concat**） |
| ONNX 输出名 | `/model.22/Mul_2_output_0`（boxes `[1,4,8400]`）+ `/model.22/Sigmoid_output_0`（scores `[1,1,8400]`） |
| config | `target_platform='rk3588'`、`mean_values=[[0,0,0]]`、`std_values=[[255,255,255]]` |
| build | `do_quantization=True`，其余全默认（w8a8 / normal / optimization_level 默认 1） |
| 校准集 | 300 张，每个视频各 75 张 |
| 产出 | `models/best_detrac_v7.rknn`，**6,482,288 字节** |

### ★ 为什么不能用 Split 拆输出头

原 ONNX 的 `output0` 是 `[1,5,8400]`，末端一个 Concat 把坐标（0~640）和分数（0~1）拼在同一个张量里。INT8 是每个张量一套量化参数，量程被坐标撑到 640 后分数全被舍成 0，所以必须拆。

**拆法有两种，只有一种能用**：

| 拆法 | 图结构 | 结果 |
|---|---|---|
| ❌ 在 Concat **后面**加 Split | `Concat_3 → output0 → Split → boxes/scores` | 1.6.0：scores 的 8400 个 int8 全是 `-128`（= zp = 0.0）；2.5.0：分数正常但框塌（float 均值 330 vs 正常的 182） |
| ✅ **删掉** Concat | `Mul_2 → boxes`、`Sigmoid → scores` 直接当输出 | 1.6.0 转出来正常 |

原因：RKNN 编译器会把挂在输出上的 Split 融合掉，两个版本各有各的融错法。删掉 Concat 就没有可融的东西。

**正确做法**（`tools/make_head3.py`）：

```python
m.graph.node.remove(concat)          # concat.name = '/model.22/Concat_3'
del m.graph.output[:]
m.graph.output.extend([
    helper.make_tensor_value_info(boxes,  TensorProto.FLOAT, [1, 4, 8400]),
    helper.make_tensor_value_info(scores, TensorProto.FLOAT, [1, 1, 8400]),
])
```

脚本自带校验：把两路输出拼回去和原模型 `output0` 逐元素比，最大差值必须是 0。

### 复现

```bash
# WSL
conda activate rknn16
python <Python版仓库>/tools/make_head3.py     # 生成+校验 ONNX
python <Python版仓库>/tools/convert_head3.py  # 转 rknn
```

### 试过但没用的（别再重复）

- `quantized_dtype`、`hybrid_level`、`custom_string`、`quantized_input_type` —— 都是猜的，无效
- `mmse` / `kl_divergence` —— 只是换裁剪范围，1 像素级收益；mmse 还容易 OOM
- 2.5.0 的 `disable_rules=['seperate_conv_rs_split']` —— 只对 2.5.0，1.6.0 没这个参数
- `optimization_level=3` —— 把 DFL 的归一化融没了，坐标算出天文数字（`cx = -2.18e33`）

想再提精度，按这个顺序：更好的校准集 → 混合量化（部分层 fp16）→ 直接上 fp16 模型 → QAT。

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

| 输出 | 合格表现 |
|---|---|
| boxes float | min 0、max ≈630、**均值 ≈182**（塌了会变成 330 左右） |
| scores int8 原始值 | 8400 个里约 **60 个非零、9 个 >0.5** |
| scores float | max ≈0.78~0.80 |

**一眼判据**：看 `[OUT1 int8]` 那行的 `==zp(-128)`——**8340 左右 = 好；8400 = 全 0，模型坏了**。

（8400 个候选框里绝大部分本来就是背景，所以"几乎全是 -128"是正常的。）

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

两个可执行文件共用 `config.cpp` / `counter.cpp` / `tracker.cpp` / `detector.cpp`，
网页版只多一个 `web_server.cpp`（HTTP + MJPEG 推流）。计数、跟踪、推理一行都没复制。

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

## 十三、已知问题

1. **计数漏（主要矛盾）**：转弯车经常漏计。可疑点见 `counter.cpp` 的 `near_segment` 和 `count_dir`
2. **框轻微向左上偏 1~2 px**：INT8 量化固有（步长 2.5 px），换 FP16 能解决但慢一倍
3. **框轻微抖动**：INT8 量化噪声 + 卡尔曼。已删掉多余的 EMA（2026-11-01），只保留卡尔曼一层平滑
4. **端口被占时仍打印「服务已启动」**：`main.cpp` 无条件打印，`web_server.cpp` 其实已经报错了
5. **网页参数只存内存**，不写回 json，重启即恢复

---

## 十四、还没做

- 计数漏的修复（当前最优先）
- 交叉编译到 aarch64（现在是在板子上直接编）
- 多路并行 + 分屏
- RTSP 实时流（现在只吃 mp4）
- 事件和计数落库（现在只在内存留 200 条）
- 人工真值全覆盖（目前只有 test3 的 B 线）

### 下一步：多路实时检测

目标：一台设备同时接入 N 路监控流，网页分屏看，点某一路放大，各路独立计数、独立参数。

**1. 视频源抽象**（最小改动，收益最大）

`Session` 里 `cv::VideoCapture` 的来源现在写死是配置里的 mp4 路径。改成三种都认：

```json
"video": "rtsp://admin:pass@192.168.1.64:554/Streaming/Channels/101"
"video": "0"                                  // 本地 USB 摄像头
"video": "data/videos/test1.mp4"              // 保持现状
```

`cv::VideoCapture` 本身三者通吃，判断一下字符串是不是纯数字、是不是带 `://` 就行。
真正的差别在**读不到帧怎么办**：mp4 读完了就结束，RTSP 断流要重连，不能把会话销毁。

**2. 多会话管理**

把 `std::unique_ptr<Session>` 换成 `std::vector<std::unique_ptr<Session>>`，
每一路一个线程跑自己的主循环，`WebServer` 仍然只有一个（多路共用一个端口）。

`Detector` 必须**每路一份**：ONNX Runtime 的 `Session` 不是线程安全的，
RKNN 的上下文也建议一路一个。代价是内存。

**3. 接口加路号**

```
GET /api/start?ch=0&config=test1
GET /stream?ch=0
GET /api/state            → {"channels":[{...},{...}]}
```

**4. 带宽要重新算**

现在一路 1080p@20fps 的 MJPEG 约 3 MB/s，本地看没问题。4 路就是 12 MB/s，
浏览器解码和网络都吃不消。要么降规格（每路 640 宽 / 10 fps，约 0.5 MB/s），
要么换架构（前端拉子码流，分析服务只推坐标 JSON 叠框）。

**5. 前端**

从"单画面 + 右侧面板"改成"分屏网格 + 单路详情"：
- 总览页：2×2 / 3×3 网格，每格一个 `<img src="/stream?ch=N">`
- 点某一格 → 进单路详情页，就是现在这个页面，多一个"返回"
