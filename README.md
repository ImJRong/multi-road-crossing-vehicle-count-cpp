# 多路口车流量统计系统 · 板端 C++ 实现

**Multi-Road Crossing Vehicle Counting System (Board Side)**

## 一、项目简介

本项目面向城市多路口监控场景，基于 YOLOv8 单类车辆检测与 ByteTrack 多目标跟踪，实现监控视频中车辆的实时检测、轨迹编号、穿线计数与分方向统计，并部署于 RK3588 NPU 边缘计算平台。

**项目背景**：路口车流量调查长期依赖人工计数或线圈检测器，前者成本高、后者需要破路施工。基于视觉的方案可直接复用既有监控摄像头，但受限于边缘设备算力与定点量化精度。

**解决的核心问题**：在多路视频并行处理的约束下，同时保证检测框定位精度与过线计数准确率。项目通过改写检测头输出、把 DFL 距离换算交由 CPU 完成，将 INT8 量化导致的框中心偏移从 6.75 px 降至 0.03 px，断号数量下降 34.3%。

**应用场景**：路口车流量调查、交通信号配时优化、固定机位交通流统计、边缘侧视频结构化。

**核心技术栈**：C++17 / OpenCV / RKNN Runtime / ONNX Runtime / YOLOv8 / ByteTrack。

## 二、技术栈

| 分类 | 内容 |
|---|---|
| 开发语言 | C++17 |
| 构建工具 | CMake 3.22.1、GNU Make |
| 推理框架 | RKNN Runtime 2.5.0（板端）、ONNX Runtime 1.31.0（PC 端） |
| 视觉库 | OpenCV 4.5.4 |
| 算法 | YOLOv8n 单类检测（`vehicle`）+ 自实现 ByteTrack（卡尔曼滤波 + 匈牙利分配 + 两阶段关联） |
| 模型转换 | rknn-toolkit2 1.6.0+81f21f4d |
| 前端 | 原生 HTML/CSS/JavaScript，MJPEG 推流，不依赖任何 CDN 与第三方库 |
| 数据格式 | 配置 JSON、结果 CSV、抓拍 JPG |

## 三、系统架构

系统采用分层架构，自下而上分为数据层、逻辑层、服务层、交互层。各层之间通过明确的接口通信，推理后端可整体替换而不影响上层代码。

```
┌────────────────────────────────────────────────────────────┐
│ 交互层  web/index.html（单路）  web/quad.html（四路）      │
├──────────────────────────────┬─────────────────────────────┤
                               │ HTTP / MJPEG
├──────────────────────────────┴─────────────────────────────┤
│ 服务层  web_server.cpp                                     │
│          HTTP 路由 / MJPEG 推流 / 状态 JSON / 控制         │
├──────────────────────────────┬─────────────────────────────┤
                               │
├──────────────────────────────┴─────────────────────────────┤
│ 逻辑层  main.cpp              命令行：配置→推理→跟踪→计数  │
│         web_main_single.cpp   单路会话管理                 │
│         web_main_quad.cpp     四路并行 + 每路独立参数      │
│         rknn_detector.cpp     RKNN 推理后端                │
│         detector.cpp          ONNX Runtime 推理后端        │
│         tracker.cpp           ByteTrack 跟踪               │
│         counter.cpp           穿线计数                     │
├──────────────────────────────┬─────────────────────────────┤
                               │
├──────────────────────────────┴─────────────────────────────┤
│ 数据层  config.cpp / mini_json.hpp   配置解析              │
│         recorder.hpp                过线记录落盘           │
│         models/*.rknn               量化后模型             │
└────────────────────────────────────────────────────────────┘
```

**核心模块划分**

| 层 | 模块 | 职责 |
|---|---|---|
| 交互层 | `web/index.html`、`web/quad.html` | 单路监控台、四路分屏页；CSS/JS 内嵌，零外部依赖 |
| 服务层 | `web_server.cpp/.hpp` | POSIX socket 自实现 HTTP 服务，MJPEG 推流、状态轮询、参数下发 |
| 逻辑层 | `main.cpp` | 命令行批处理：配置 → ROI 裁剪 → 推理 → 跟踪 → 计数 → 绘制 |
| 逻辑层 | `web_main_single.cpp` / `web_main_quad.cpp` | 常驻进程的会话管理，单路 / 四路 |
| 逻辑层 | `rknn_detector.cpp` / `detector.cpp` | 推理后端封装，前后处理逻辑完全一致，仅后端不同 |
| 逻辑层 | `tracker.cpp` + `kalman_track.hpp` | ByteTrack 两阶段关联 |
| 逻辑层 | `counter.cpp` | 线段侧别判定、死区、换边计数 |
| 数据层 | `config.cpp` / `mini_json.hpp` | `configs/*.json` 解析，不依赖第三方 JSON 库 |
| 数据层 | `recorder.hpp` | 过线记录写 CSV |

**设计约束**：换推理后端只改一个 `.cpp`，`main.cpp` 一行不动。这是把推理封装成 `Detector` 类的唯一目的。

## 四、环境依赖

### 4.1 板端（部署目标，实测环境）

| 项 | 版本 / 值 |
|---|---|
| 硬件 | Orange Pi 5 Plus（RK3588，4×Cortex-A76 + 4×Cortex-A55，5.8 GiB） |
| 操作系统 | Ubuntu 22.04.5 LTS（Jammy），内核 `5.10.160-rockchip-rk3588` |
| 编译器 | g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 |
| 构建工具 | CMake 3.22.1 |
| OpenCV | 4.5.4（系统包 `libopencv-dev`） |
| NPU 运行库 | `librknnrt.so` 2.5.0（`/usr/lib/librknnrt.so`） |
| NPU 驱动 | 0.9.6 |

> 板端不装 ONNX Runtime，只编 RKNN 后端。

### 4.2 PC 端（调试 / 模型转换）

| 项 | 版本 / 值 |
|---|---|
| 操作系统 | Ubuntu 26.04.1 LTS（WSL2） |
| 编译器 | g++ 15.2.0 |
| 构建工具 | CMake 4.2.3 |
| OpenCV | 4.10.0（系统包 `libopencv-dev`） |
| ONNX Runtime | 1.31.0（`~/onnxruntime-linux-x64-1.31.0`） |
| conda | Miniforge3（`~/miniforge3`） |
| 模型转换 | rknn-toolkit2 **1.6.0+81f21f4d**，conda 环境 `rknn16`，Python 3.10.22 |
| 对照环境 | rknn-toolkit2 2.5.0，conda 环境 `rknn` / `rknn20`，Python 3.10.22 |

> WSL 侧只用于转模型与对拍，不参与板端编译。转换脚本对工具链版本不敏感，
> 但 rknn-toolkit2 必须严格为 1.6.0。

## 五、项目结构

```
traffic_count/
├── CMakeLists.txt          主构建脚本（命令行版 + rknn_probe / multi_eval / pipeline_bench）
├── web.cmake               网页版构建脚本（traffic_web_single / traffic_web_quad）
├── README.md               本文件
├── LICENSE                 学术使用许可（Academic Use Only）
├── .gitattributes          统一行尾（仓库内 LF）
├── docs/
│   ├── 技术文档.md         设计推导、实测数据、算法细节、踩坑记录
│   └── 代码走读.md         逐文件逐函数讲解：每个函数干什么、为什么这么写
├── configs/                每个监控点一份 JSON（不入库，随部署生成）
│   └── test1.json ~ test4.json
├── data/
│   └── videos/             原始路口视频 test1~test4（不入库）
├── models/
│   ├── best_detrac.onnx    ONNX 模型（中转用）
│   └── best_detrac.rknn    板端模型（head4，INT8）
├── src/                    核心源码，共 4983 行
│   ├── main.cpp            命令行入口
│   ├── web_main_single.cpp 网页版入口（单路）
│   ├── web_main_quad.cpp   网页版入口（四路）
│   ├── web_server.cpp/.hpp HTTP 服务 + MJPEG
│   ├── rknn_detector.cpp/.hpp RKNN 推理后端
│   ├── detector.cpp/.hpp   ONNX Runtime 推理后端
│   ├── tracker.cpp/.hpp    ByteTrack 跟踪
│   ├── kalman_track.hpp    卡尔曼滤波
│   ├── counter.cpp/.hpp    穿线计数
│   ├── config.cpp/.hpp     配置解析
│   ├── mini_json.hpp       自写 JSON 解析
│   ├── recorder.hpp        CSV 记录
│   ├── detection.hpp       检测框结构体（两个后端共用）
│   ├── rknn_probe.cpp      模型体检工具
│   ├── multi_eval.cpp      多路批量评测工具
│   └── pipeline_bench.cpp  流水线性能测试工具
├── web/
│   ├── index.html          单路监控台
│   └── quad.html           四路分屏
└── tools/
    └── convert_v6.py       模型转换脚本
```

**配套文档**

| 想知道什么 | 去哪 |
|---|---|
| 设计推导、实测原始数据、算法细节、踩坑记录 | [docs/技术文档.md](docs/技术文档.md) |
| 代码怎么读、每个函数在干什么 | [docs/代码走读.md](docs/代码走读.md) |
| 框偏移的原理推导（11 张图） | Python 版仓库 [docs/head4_原理详解.md](https://github.com/ImJRong/multi-road-crossing-vehicle-count/blob/main/docs/head4_%E5%8E%9F%E7%90%86%E8%AF%A6%E8%A7%A3.md) |
| 完整踩坑记录与交接文档 | Python 版仓库 [MEMORY.md](https://github.com/ImJRong/multi-road-crossing-vehicle-count/blob/main/MEMORY.md) |
| Python 原型（同类算法的参照实现） | [multi-road-crossing-vehicle-count](https://github.com/ImJRong/multi-road-crossing-vehicle-count) |

## 六、编译 / 运行 / 部署

### 6.1 板端编译（RKNN 后端）

```bash
# 1. 取代码
git clone https://github.com/ImJRong/multi-road-crossing-vehicle-count-cpp.git ~/code
cd ~/code

# 2. 确认依赖存在
ls /usr/lib/librknnrt.so          # 必须有
pkg-config --modversion opencv4   # 必须返回 4.x

# 3. 首次编译（RKNN 头文件在 src/ 下，运行库在 /usr/lib）
cmake . -DUSE_RKNN=ON -DRKNN_ROOT=/usr -DTRAFFIC_PROJECT_ROOT=~/code
make -j4
```

编出以下可执行文件：

| 目标 | 用途 |
|---|---|
| `traffic_count` | 命令行版，跑完退出 |
| `traffic_web_single` | 网页版，单路常驻 |
| `traffic_web_quad` | 网页版，四路分屏常驻 |
| `rknn_probe` | 模型体检 |
| `multi_eval` | 多路批量评测 |
| `pipeline_bench` | 流水线性能测试 |

也可以只编其中一个：`make traffic_web_quad -j4`。

> **改了 `CMakeLists.txt` 里的路径必须删缓存**：`set(... CACHE ...)` 不会覆盖缓存旧值，
> 不删会一直用老路径。执行 `rm -f CMakeCache.txt` 后重新 cmake。

### 6.2 准备数据与配置

配置读 `$TRAFFIC_PROJECT_ROOT/configs/testN.json`，视频路径写在配置里。

```bash
mkdir -p configs data/videos
# 把 test1.mp4 ~ test4.mp4 放进 data/videos/
# 把对应的 test1.json ~ test4.json 放进 configs/
```

配置文件结构见 §七 模块说明中的「配置模块」。

### 6.3 运行（三种方式）

**方式一：命令行批处理**

```bash
./traffic_count test4      # 跑 test4 配置，跑完退出
./traffic_count            # 不带参数，列出现有配置
```

**方式二：单路网页监控**

```bash
./traffic_web_single 8080
# 浏览器打开 http://<板子IP>:8080，选配置点「开始」
```

**方式三：四路分屏（推荐用于多路口部署）**

```bash
# 位置参数: [端口] 配置1 配置2 配置3 配置4 [每路fps]
setsid ./traffic_web_quad 8081 test1 test2 test3 test4 20 > /tmp/quad.log 2>&1 < /dev/null &
# 浏览器打开 http://<板子IP>:8081
```

不写端口默认 8081，不写配置名默认 test1~test4，不写 fps 默认 20。

> 后台常驻必须用 `setsid`。`nohup ... &` 在非交互 SSH shell 里会被回收。

**停止服务**

```bash
pkill -f "[t]raffic_web_quad"
```

> `[t]` 是必需的：不加方括号时 `pkill -f` 会把执行这条命令的 shell 自己也匹配上一起杀掉。

### 6.4 部署注意事项

| 项 | 说明 |
|---|---|
| CPU 调频 | 默认 `ondemand` 下 15.6 fps，`performance` 下 20.6 fps。重启失效，需重设：`for p in /sys/devices/system/cpu/cpufreq/policy*; do echo performance \| sudo tee $p/scaling_governor; done` |
| 输出模式 | `configs/*.json` 的 `output` 设 `"none"` 才是纯推理测速；设 `"show"` 在无显示器的板子上必崩（GTK 初始化失败） |
| 端口占用 | 端口被占时程序会打印「服务已启动」但实际未监听。起服务前先 `pkill` 旧进程 |
| 内存 | 四路并行约占用 1.0 GiB / 5.8 GiB，余量充足 |

## 七、核心功能模块

### 1. 推理模块（`rknn_detector.cpp` / `detector.cpp`）

**作用**：把一帧图像送进模型，输出检测框与置信度。

**实现原理**：letterbox 缩放到 640×640 并保持长宽比 → 归一化到 0~1 → RKNN 推理 → 对两路输出做后处理（分数过滤 → NMS）→ 坐标还原到原图。

**关键技术点**：模型按输出张量名是否含 `"dfl"` 自动识别模型类型，因此换 head3 / head4 模型无需重新编译。

```cpp
impl_->boxes_is_dist = std::string(a.name).find("dfl") != std::string::npos;
```

含 `dfl` 时输出的是 DFL 距离（0~15），走 `anchor ± 距离 × stride` 换算；不含时输出的是归一化坐标，直接取用。

### 2. 跟踪模块（`tracker.cpp`）

**作用**：给每辆车分配稳定编号，跨帧关联同一目标。

**实现原理**：自实现 ByteTrack，每帧执行两阶段匈牙利关联。卡尔曼滤波预测位置 → 所有轨迹与高分框（≥0.25）匹配，代价 `1-IoU`、阈值 0.8 → 未匹配轨迹再与低分框（0.1~0.25）匹配，阈值 0.5 → 高分框无人认领则新建轨迹，低分框无人认领则丢弃 → 两轮都未匹配的 `lost+1`，超过 30 帧删除。

**关键技术点**：输出框直接取卡尔曼滤波后的状态，不叠加 EMA —— 叠加属于重复平滑，会额外引入 1~2 帧滞后。低分框不允许新建轨迹，否则树影、反光会在一帧内产生十几个假 ID。

### 3. 计数模块（`counter.cpp`）

**作用**：判定车辆是否穿过计数线，并按方向累计。

**实现原理**：每帧对每辆车调用一次 `update()`。用叉积判断点在线段的哪一侧，用投影比例判断点是否落在线段范围内，符号变化即视为换边。设置死区（`margin` 像素）防止车辆压线抖动时反复计数。

**关键技术点**：`count_dir` 支持只统计单一方向（`"pos"` = 负侧→正侧，`"neg"` = 正侧→负侧）。单行道场景下这属于设计意图而非缺陷。`retain(live_ids)` 每帧清理已被跟踪器删除的车辆编号，否则长时间运行会累积数十万条状态。

### 4. 网页服务模块（`web_server.cpp`）

**作用**：提供 HTTP 服务，推流实时画面，接收控制命令。

**实现原理**：POSIX socket 手写 HTTP 服务，不依赖第三方库。一个 accept 线程，每个浏览器客户端一个 detached 线程；推流线程通过条件变量等待新帧，不空转。控制指令通过 `atomic` 变量传给推理主循环。

**关键技术点**：服务端绘制完成后再编码推流（不是前端叠图），因此前端只需展示 JPEG。客户端提前关闭页面时忽略 `SIGPIPE`，避免整个进程被打死。

### 5. 四路并行模块（`web_main_quad.cpp`）

**作用**：单块板卡同时跑四路视频，网页 2×2 分屏查看。

**实现原理**：每路 2 个推理线程（`kInferPerCam = 2`），共 8 个，按 `(路号 × 2 + 槽位) % 3` 轮流绑定到 NPU 的三个核心；另有每路 1 个读帧线程和 1 个跟踪线程。四路共用一个 `WebServer` 与同一个端口，接口通过 `cam=` 参数区分路号。

**关键技术点**：参数面板初值来自各自配置文件的 `detect.conf/iou`（`WebServer::init_params()`），网页上修改只写内存，配置文件一个字节都不动，进程重启即恢复配置值。

### 6. 配置模块（`config.cpp` / `mini_json.hpp`）

**作用**：解析 `configs/*.json`，为其余模块提供参数。

**配置结构**

```json
{
  "video": "data/videos/test1.mp4",
  "max_frames": 400,
  "detect": { "imgsz": 640, "conf": 0.3, "iou": 0.3 },
  "roi": [[983,90],[278,90],[278,540],[983,540]],
  "lines": [
    { "name": "A", "a": [390,140], "b": [695,140],
      "label_at": [406,130], "margin": 10, "count_dir": "neg" }
  ],
  "draw": { "box": true, "id": true, "line": true },
  "output": "none"
}
```

**关键技术点**：JSON 解析器为自写实现（`mini_json.hpp`，223 行），不引入第三方依赖，交叉编译时无需额外处理。

### 7. 辅助工具模块

| 工具 | 作用 |
|---|---|
| `rknn_probe` | 打印模型输入输出张量的形状、类型、`scale`、`zp`，并真实跑一帧后原样摊开输出值。换新模型先跑它 |
| `multi_eval` | 多路并行跑完整视频，输出计数统计 CSV，用于量化比较两个模型 |
| `pipeline_bench` | 并行推理 + 串行跟踪的流水线压测，支持 `--realtime N` 模拟实时摄像头节奏 |

## 八、关键技术难点与解决方案

### 难点一：INT8 量化导致检测框整体向左上偏移

**问题**：板端检测框相比 PC 端 FP32 结果整体向左上偏移，且车越小偏移越大，实测中心偏移 6.75 px（box1 左 6.4 / 上 14.5 px，box3 左 10 / 上 20 px）。

**成因**：RKNN 的量化为每个张量按 `范围 / 256` 分配刻度。原模型输出坐标（0~640），刻度被撑到 2.5017；而 DFL 内部的距离值只有 0~15，在粗刻度下仅剩 6 个有效量化台阶，精度被完全稀释。

完整误差链条为六步逐级放大 53 倍：

```
DFL 距离 0~11.97        刻度 0.047
anchor − 左距离          刻度 0.325   ← 关键：大数(0.5~79.5)减小数(0~15)，小数值精度被稀释
anchor + 右距离          刻度 0.327
两者相加 ÷ 2             刻度 0.625
两者相减 = 宽高          刻度 0.084
拼 4 通道 × stride       刻度 2.502
```

**解决方案**：把模型输出改成 DFL 距离本身，坐标换算移到 CPU 用 float 完成。

- 从 ONNX 图中删除 DFL 之后的 9 个换算节点（Slice×2 / Sub×2 / Add×2 / Div / Concat / Mul），将 `/model.22/dfl/Reshape_1_output_0` 直接设为图输出。这 9 个节点不含任何权重。
- 距离成为独立输出后，RKNN 按其真实值域分配刻度 `0.046943`，精度提升 53 倍。
- C++ 侧换算公式：`x1 = (anchor_x − 左距离) × stride`，`y1 = (anchor_y − 上距离) × stride`，右边界与下边界同理用加号。

**效果**：框中心偏移 6.75 px → 0.03 px，模型权重未改、量化方式未改、推理速度未变。

验证分三道关：ONNX 输出与原模型逐元素差值为 0；numpy 重写换算公式与原模型 boxes 差值为 0；板端实测 10 帧，左右上下偏移均 ≤ 0.04 px。

### 难点二：拆输出头时分数被全部量化成 0

**问题**：原模型 `output0` 是 `[1,5,8400]`，坐标（0~640）与分数（0~1）拼在同一张量里，INT8 每个张量只有一套量化参数，分数被坐标范围撑大后全部舍入为 0。

**尝试过但失败的方案**：在 Concat 之后加 Split 拆成两路。RKNN 编译器会把挂在输出上的 Split 融合掉 —— 1.6.0 下 8400 个分数全为 `-128`（即 zp，对应 0.0）；2.5.0 下分数正常但框塌成一片。

**解决方案**：直接删掉末端的 Concat，把 `Mul_2`（坐标）和 `Sigmoid`（分数）各自作为图输出，没有可被融合的节点。

**验证手段**：生成脚本把两路输出拼回去与原模型 `output0` 逐元素比较，最大差值必须为 0。

### 难点三：rknn-toolkit2 2.5.0 转出的模型宽高放大 13 倍

**问题**：同一份 ONNX、同一批校准图、同一套配置，只换工具版本，2.5.0 转出的模型框宽高放大约 13 倍，4 个输出通道的实测范围全部顶到 637.92（= 255 × 2.5 的饱和值），而正常模型四个通道各不相同。

**排查过程**：先用基线 INT8 与 `auto_hybrid`（自动混合量化）各转一版，两者均坏；再通过冒烟测试确认 `auto_hybrid` 开关本身工作正常（阈值变化确实改变了模型体积与输入 dtype）。逐层比对后确认：2.5.0 把 DFL 距离（0~15）与坐标（0~640）塞进同一套量化参数，属于工具链自身的缺陷，不是混合量化功能的问题。

**解决方案**：转换固定使用 rknn-toolkit2 **1.6.0+81f21f4d**，conda 环境 `rknn16`。2.5.0 仅用于对照实验。

### 难点四：转弯车辆漏计

**问题**：车辆转弯时计数线判定失效，出现漏计。

**排查与结论**：改用 head4 模型后，12135 帧测试数据上轨迹断号从 2559 降至 1680（下降 34.3%），计数净增 12 次且无任何一条线下降。说明框抖动导致的编号断裂是一个成因，但只解释了约三分之一的漏计。

**当前状态**：剩余成因未进一步查证。分析认为投入产出比不高，已标记收尾。

### 难点五：四路并行抢占单核导致吞吐不升

**问题**：四路视频同时推理时，吞吐不升反降，NPU 只有一个核心在工作。

**解决方案**：`RknnDetector::set_core(int core_id)` 按路号绑定 NPU 核心，四路的 8 个推理线程按 `(路号 × 2 + 槽位) % 3` 分散到三个核心。

**实测效果**：1/2/3/4 路并行分别为 17.3 / 37.9 / 58.3 / 71.2 fps，接近线性扩展；四路各 20 fps 稳定运行，单帧推理 40~60 ms。

### 难点六：四路分屏页面越看越卡

**问题**：四路分屏页面打开一段时间后明显卡顿。

**成因**：4 条 MJPEG 常驻推流单条约 500 KB/s（缩略图宽 480 px），四条合计约 2 MB/s。浏览器对同一主机只保持 6 个并发连接，4 条流占满后状态轮询只能排队。同时服务端在每帧都执行 `cv::imencode`，四路并行时 CPU 占用约 487%。

**当前状态**：已定位未解决。候选方案为降低缩略图帧率或分辨率，或把多次轮询合并为一次请求。

## 九、版本迭代记录

| 版本 | 日期 | 内容 |
|---|---|---|
| v1.0.0 | 2026-11-06 | 完成四路并行分屏网页版（`traffic_web_quad`）；参数初值改为从配置文件读取；补齐 README 工程文档 |
| v0.9.0 | 2026-11-02 | 修复框偏移：模型输出改为 DFL 距离，中心偏移 6.75 px → 0.03 px；修复计数漏（断号下降 34.3%） |
| v0.8.0 | 2026-11-01 | 修复计数记录只增不删的内存泄漏；删除跟踪输出上的多余 EMA 平滑；模型转换配方定型 |
| v0.7.0 | 2026-10-30 | 板端适配：CPU 调频、输出模式、端口处理 |
| v0.6.0 | 2026-10-28 | 实现基础网页监控台（`traffic_web_single`）：MJPEG 推流、图层开关、参数调整、抓拍记录 |
| v0.5.0 | 2026-10-22 | INT8 量化精度验证：拆分输出头，完成 FP32 / FP16 / INT8 三方对比（mAP50 下降 3.9 点） |
| v0.4.0 | 2026-10-21 | 实现 ByteTrack 两阶段关联，跟踪精度与 ultralytics 对齐 |
| v0.3.0 | 2026-10-19 | 接入 RKNN 推理后端，新增过线记录 CSV 导出 |
| v0.2.0 | 2026-10-18 | C++ 版初版：ROI 裁剪、IoU 跟踪、穿线计数 |

## 十、作者与声明

| 项 | 内容 |
|---|---|
| 作者 | jiarong（RONGG） |
| 仓库 | https://github.com/ImJRong/multi-road-crossing-vehicle-count-cpp |
| 相关仓库 | https://github.com/ImJRong/multi-road-crossing-vehicle-count （Python 原型与模型转换） |
| 日期 | 2026-11-06 |

**License**：仅供学术研究（Academic Use Only）。完整条款见 [LICENSE](LICENSE)。
未经作者书面许可，不得用于商业用途、不得再分发与对外提供服务。

模型基于 UA-DETRAC 数据集微调，数据集版权归原作者所有。YOLOv8 采用 Ultralytics 开源实现，遵循其 AGPL-3.0 许可。
