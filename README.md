# 多路口车流量统计系统 · 板端 C++ 实现

**Multi-Road Crossing Vehicle Counting System (Board Side)**

## 一、项目简介

路口车流量调查目前主要靠两种办法：人工数，或者埋地感线圈。人工贵，线圈要破路。摄像头本来就在杆子上，直接拿来用最省事，难的是边缘设备的算力和定点量化的精度。

这个项目把 YOLOv8 单车类检测和 ByteTrack 跟踪放到 RK3588 的 NPU 上跑，做实时检测、轨迹编号、穿线计数和分方向统计。四路视频并行时，框的定位精度和过线计数的准确率要同时守住——INT8 量化会让框整体往左上跑，实测中心偏移 6.75 px。最后靠改写检测头输出、把 DFL 距离的换算挪回 CPU 解决，偏移降到 0.03 px，轨迹断号少了 34.3%。

适用场景：路口流量调查、信号配时、固定机位交通流统计、边缘侧视频结构化。

技术栈：C++ / OpenCV / RKNN Runtime / ONNX Runtime / YOLOv8 / ByteTrack。

## 二、技术栈

| 分类 | 内容 |
|---|---|
| 开发语言 | C++ |
| 构建工具 | CMake 3.22.1、GNU Make |
| 推理框架 | RKNN Runtime 2.5.0（板端）、ONNX Runtime 1.31.0（PC 端） |
| 视觉库 | OpenCV 4.5.4 |
| 算法 | YOLOv8n 单车类检测（`vehicle`）+ 自实现 ByteTrack（卡尔曼滤波 + 匈牙利分配 + 两阶段关联） |
| 模型转换 | rknn-toolkit2 1.6.0+81f21f4d |
| 前端 | 原生 HTML/CSS/JavaScript，MJPEG 推流，不依赖 CDN 与第三方库 |
| 数据格式 | 配置 JSON、结果 CSV、抓拍 JPG |

## 三、系统架构

分四层：数据层、逻辑层、服务层、交互层。层与层之间只走明确的接口，推理后端可以整套换掉，上层不用动。

| 分层 | 文件 | 功能 |
| --- | --- | --- |
| 交互层 | web/index.html | 单路监控台 |
|  | web/quad.html | 四路分屏 |
| 服务层 | web_server.cpp | HTTP 路由 / MJPEG 推流 |
| 逻辑层 | main.cpp | 命令行批处理 |
|  | web_main_single.cpp | 单路会话 |
|  | web_main_quad.cpp | 四路并行 |
|  | rknn_detector.cpp | RKNN 推理 |
|  | detector.cpp | ONNX Runtime 推理 |
|  | tracker.cpp | ByteTrack 跟踪 |
|  | counter.cpp | 穿线计数 |
| 数据层 | config.cpp / mini_json.hpp | 配置解析 |
|  | recorder.hpp | 过线记录 |
|  | models/*.rknn | 量化模型 |


**核心模块划分**

| 层 | 模块 | 职责 |
| --- | --- | --- |
| 交互层 | `web/index.html`、`web/quad.html` | 单路监控台、四路分屏页；CSS/JS 内嵌，不引外部资源 |
| 服务层 | `web_server.cpp/.hpp` | POSIX socket 手写 HTTP，MJPEG 推流、状态轮询、参数下发 |
| 逻辑层 | `main.cpp` | 命令行批处理：配置 → ROI 裁剪 → 推理 → 跟踪 → 计数 → 绘制 |
|  | `web_main_single.cpp` / `web_main_quad.cpp` | 常驻进程的会话管理，单路与四路 |
|  | `rknn_detector.cpp` / `detector.cpp` | 两个推理后端，前后处理完全一致，只有底层库不同 |
|  | `tracker.cpp` + `kalman_track.hpp` | ByteTrack 两阶段关联 |
|  | `counter.cpp` | 判断点在线段哪一侧、死区、换边计数 |
| 数据层 | `config.cpp` / `mini_json.hpp` | 解析 `configs/*.json`，不依赖第三方 JSON 库 |
|  | `recorder.hpp` | 过线记录写 CSV |

换推理后端只改一个detector，`main.cpp` 不用动。

## 四、环境依赖

### 4.1 板端（部署目标，实测环境）

| 项 | 版本 / 值 |
|---|---|
| 硬件 | Orange Pi 5 Plus（RK3588，6G+64G） |
| 操作系统 | Ubuntu 22.04.5 LTS（Jammy），内核 `5.10.160-rockchip-rk3588` |
| 编译器 | g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 |
| 构建工具 | CMake 3.22.1 |
| OpenCV | 4.5.4（系统包 `libopencv-dev`） |
| NPU 运行库 | `librknnrt.so` 2.5.0（`/usr/lib/librknnrt.so`） |
| NPU 驱动 | 0.9.6 |

板端不装 ONNX Runtime，只编 RKNN 后端。

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

WSL 侧只用来转模型和做对拍，不参与板端编译。

## 五、项目结构

```
traffic_count/
├── CMakeLists.txt          主构建脚本（命令行版 + rknn_probe / multi_eval / pipeline_bench）
├── web.cmake               网页版构建脚本（traffic_web_single / traffic_web_quad）
├── README.md               项目说明
├── .gitattributes          统一行尾（仓库内 LF）
├── docs/                   文档，见本节下方「配套文档」
│   ├── head4_原理详解.md   框偏移原理推导（11 节 11 图）
│   ├── 技术文档.md         设计推导、实测数据、算法细节
│   ├── 代码走读.md         逐文件逐函数讲解：每个函数干什么、为什么这么写
│   ├── 实操手册.md         训练数据准备、参数标定、量化实验
│   ├── 视频推流操作手册.md 用 PC 的 MP4 伪装 RTSP 摄像头，板子拉流
│   ├── 模型转换.md         RKNN 转换配方、验证方法、量化刻度
│   ├── 性能数据.md         单帧耗时构成、吞吐、CPU 调频
│   ├── figures/            原理图 11 张
│   └── screenshots/        误差实测截图 3 张
├── configs/                每个监控点一份 JSON，随仓库提供
│   └── test1.json ~ test4.json
├── data/
│   └── videos/             原始路口视频 test1~test4（不入库，需自备）
├── models/
│   ├── best_detrac.onnx    ONNX 模型
│   └── best_detrac.rknn    板端模型（INT8，5.7 MB）
├── src/                    
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
└── output/                 运行产物：结果视频、CSV、抓拍图
```


## 六、编译 / 运行 / 部署

### 6.1 板端编译（RKNN 后端）

板子没有外网，代码从 PC 侧传过去。

```bash
# 1. 把代码放到 ~/code
#    从 WSL 仓库同步（代码正本在 WSL 仓库 ~/traffic_count）
#    或在 PC 上执行: scp -r traffic_count/* orangepi@192.168.1.105:~/code/
cd ~/code

# 2. 确认依赖存在
ls /usr/lib/librknnrt.so          
pkg-config --modversion opencv4   # 必须返回 4.x

# 3. 首次编译（RKNN 头文件在 src/ 下，运行库在 /usr/lib）
cmake . -DUSE_RKNN=ON -DRKNN_ROOT=/usr -DTRAFFIC_PROJECT_ROOT=~/code
make -j4
```

编出来六个可执行文件：

| 目标 | 用途 |
|---|---|
| `traffic_count` | 命令行版，跑完退出 |
| `traffic_web_single` | 网页版，单路常驻 |
| `traffic_web_quad` | 网页版，四路分屏常驻 |
| `rknn_probe` | 模型体检 |
| `multi_eval` | 多路批量评测 |
| `pipeline_bench` | 流水线性能测试 |


改了 `CMakeLists.txt` 里的路径记得删缓存。`set(... CACHE ...)` 不覆盖已有的缓存值，不删就一直用老路径，表现是编译通过但运行时找不到模型或配置。删完重新 cmake：

```bash
rm -f CMakeCache.txt
```

### 6.2 准备数据与配置

配置读 `$TRAFFIC_PROJECT_ROOT/configs/testN.json`，视频路径写在配置里。`configs/test1.json` ～ `test4.json` 随仓库提供，直接用，只需要自备视频：

```bash
mkdir -p data/videos
# 把 test1.mp4 ~ test4.mp4 放进 data/videos/
```

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

**方式三：四路分屏（多路口部署用这个）**

```bash
./traffic_web_quad 8081
# 浏览器打开 http://<板子IP>:8081
```

只给端口就行。配置名默认取 `test1`～`test4`，帧率默认跟源视频走：test1 12 fps、test2 30 fps、test3 24 fps、test4 8 fps。老写法 `./traffic_web_quad 8081 test1 test2 test3 test4 20` 仍然兼容，第二个数字参数是每路帧率上限。

要后台常驻必须用 `setsid`，`nohup ... &` 在非交互 SSH shell 里会被回收。

**停止服务**

```bash
pkill -f "[t]raffic_web_quad"
```


### 6.4 部署注意事项

| 项 | 说明 |
|---|---|
| CPU 调频 | 默认 `ondemand` 下 15.6 fps，`performance` 下 20.6 fps。重启失效，需重设：`for p in /sys/devices/system/cpu/cpufreq/policy*; do echo performance \| sudo tee $p/scaling_governor; done` |
| 输出模式 | `output` 设 `"none"` 用于纯推理测速；设 `"show"` 在无显示器的板端会崩溃（GTK 初始化失败） |
| 端口占用 | 端口被占时程序仍会打印「服务已启动」，但实际没监听。起服务前先 `pkill` 旧进程 |
| 内存 | 四路并行约占用 1.0 GiB / 5.8 GiB |

### 6.5 接 RTSP 视频流（网页换源）

板子没有摄像头时，可以用 PC 把 MP4 伪装成 RTSP 摄像头。推流侧的搭建与改法见
[docs/视频推流操作手册.md](docs/视频推流操作手册.md)，这里只讲板子端怎么用。

启动顺序不能反：

1. PC 上双击 `D:\tools\mediamtx\mediamtx.exe`，黑窗口不要关
2. 板子上 `cd ~/code && ./traffic_web_quad 8081`
3. 浏览器打开 `http://<板子IP>:8081`

监控台顶部有「视频源」卡片，四行对应四路。

| 列 | 作用 |
|---|---|
| 路口 1～4 | 路号标签 |
| 类型 | 本地视频 / 视频流 |
| 地址 | 选本地视频时是下拉框（扫 `data/videos/`），选视频流时是输入框 |
| 配置 | 用哪份 `configs/*.json`，决定 ROI、计数线、跟踪阈值 |
| 接入 / 断开 | 起停这一路 |


板子上填的地址（`192.168.1.101` 是 PC 的局域网 IP，换网络要重新查）：

```
rtsp://192.168.1.101:8554/test1    1280x674  12 fps
rtsp://192.168.1.101:8554/test2    1280x720  30 fps
rtsp://192.168.1.101:8554/test3    1280x720  24 fps
rtsp://192.168.1.101:8554/test4    1280x960   8 fps
```

关闭：

```bash
pkill -f "[t]raffic_web_quad"
```

PC 侧：

```powershell
Get-Process mediamtx, ffmpeg -ErrorAction SilentlyContinue | Stop-Process -Force
```

## 七、核心功能模块

### 1. 推理（`rknn_detector.cpp` / `detector.cpp`）

一帧图进来，先 letterbox 到 640×640 并保持长宽比，归一化到 0～1，送进模型；两路输出各自做分数过滤和 NMS，再把坐标还原回原图。

换 head3 / head4 模型不用重新编译。程序看坐标输出张量的名字里有没有 `dfl`：

```cpp
impl_->boxes_is_dist = std::string(a.name).find("dfl") != std::string::npos;
```

有，说明吐的是 DFL 距离（0～15），按 `anchor ± 距离 × stride` 换算；没有，就是归一化坐标，直接取用。

### 2. 跟踪（`tracker.cpp`）

自实现的 ByteTrack，每帧两阶段匈牙利关联。卡尔曼先预测位置，所有轨迹跟高分框（≥ `track.high`）配一次，代价 `1-IoU`，阈值 0.8；没配上的轨迹再跟低分框（`track.low`～`track.high`）配一次，阈值收到 0.5；高分框没人认领就新建轨迹，低分框没人认领直接丢；两轮都落空的 `lost+1`，超过 30 帧删掉。

两处容易出错。输出的框直接取卡尔曼滤波后的状态，不要再叠一层 EMA——那属于重复平滑，白搭 1～2 帧滞后。低分框也不允许新建轨迹，否则树影反光能在一帧里冒出十几个假编号。

### 3. 计数（`counter.cpp`）

每帧对每辆车调一次 `update()`。叉积判断点落在线段的哪一侧，投影比例判断点是否在段内，符号翻过来就算换边。中间设一段死区（`margin` 像素），防止车压着线抖动时反复计数。

`count_dir` 可以只统计一个方向：`"pos"` 是负侧到正侧，`"neg"` 反过来。单行道场景下这是设计意图，不是缺陷。另外每帧调一次 `retain(live_ids)`，把跟踪器已经删掉的编号一并清掉，不然长时间运行会堆到几十万条状态。

### 4. 网页服务（`web_server.cpp`）

POSIX socket 手写的 HTTP 服务，没有第三方库。一个 accept 线程，每个浏览器客户端一个 detached 线程；推流线程等条件变量拿新帧，不空转。控制指令通过 `atomic` 变量交给推理主循环。

框是服务端画好再编码推流的，不是前端叠图，所以前端只负责显示 JPEG。客户端提前关页面时忽略 `SIGPIPE`，否则整个进程会被打断。

### 5. 四路并行（`web_main_quad.cpp`）

单板跑四路视频，网页 2×2 分屏。每路 2 个推理线程（`kInferPerCam = 2`），一共 8 个，用全局线程计数器轮流绑到 NPU 的三个核上；另有每路 1 个读帧线程和 1 个跟踪线程，只跑一路时推理线程提到 3 个。四路共用同一个 `WebServer` 和端口，接口用 `cam=` 区分路号。

三个地方值得记一下：

- **视频源可以热切换。** 每路的地址和配置名存在 `WebServer` 里（`source_` / `src_cfg_`），外加一个版本号 `src_gen_`。`/api/source` 只改字符串、把版本号加一，真正的线程起停由主循环轮询发现后执行（`restart_one`）。地址前缀是 `rtsp:// rtsps:// http:// https://` 的按流处理，其余当本地文件。
- **停线程必须唤醒条件变量。** `stop_one()` 把 `alive` 置 false 之后要立刻 `cv.notify_all()`，否则堵在 `cv.wait` 上（谓词只在 alive 变化时求值，可没人负责叫醒）的推理线程永远不返回，`join()` 卡死，主线程跟着卡死，整个服务不再响应换源。
- **参数面板的初值**来自各自配置文件的 `detect.conf/iou`（`WebServer::init_params()`）。网页上改只写内存，配置文件一个字节都不动，进程重启即恢复配置值。

### 6. 配置（`config.cpp` / `mini_json.hpp`）

```json
{
  "video": "data/videos/test1.mp4",
  "max_frames": 400,
  "detect": { "imgsz": 640, "conf": 0.3, "iou": 0.3 },
  "track": { "buffer": 30, "high": 0.50, "low": 0.30, "new": 0.50, "match": 0.8 },
  "roi": [[983,90],[278,90],[278,540],[983,540]],
  "lines": [
    { "name": "A", "a": [390,140], "b": [695,140],
      "label_at": [406,130], "margin": 10, "count_dir": "neg" }
  ],
  "draw": { "box": true, "id": true, "line": true },
  "output": "none"
}
```

`track` 段可以不写，不写就用 `tracker.hpp` 里的默认值（0.45 / 0.30 / 0.45 / 0.8 / 30）。注意 `low` 配得比 `detect.conf` 低没有意义——低于 `conf` 的框在检测阶段就被扔了，根本到不了跟踪器。实测依据见 `docs/技术文档.md` 第八点五节。

JSON 解析器是自写的（`mini_json.hpp`），不引第三方依赖，交叉编译时省事。

### 7. 辅助工具

| 工具 | 作用 |
|---|---|
| `rknn_probe` | 打印模型输入输出张量的形状、类型、`scale`、`zp`，再真跑一帧把输出值原样摊开。换新模型先跑它 |
| `multi_eval` | 多路并行跑完整视频，出计数统计 CSV，用来量化比较两个模型 |
| `pipeline_bench` | 并行推理 + 串行跟踪的流水线压测，支持 `--realtime N` 模拟实时摄像头节奏 |

### 8. 模型转换（离线，在 PC / WSL 侧做）

训练好的 ONNX 要转成板端能跑的 RKNN INT8，分三步。

第一步改输出头。原 YOLOv8 ONNX 输出的是最终坐标，值域 0～640。转换前把 `Concat` / `Mul` 之后的 9 个节点（Slice×2 / Sub×2 / Add×2 / Div / Concat / Mul）删掉，让 `/model.22/dfl/Reshape_1_output_0` 直接成为图输出。这样模型吐的是 DFL 距离（值域 0～15），坐标换算搬到 C++ 里做。这 9 个节点没有权重，删掉不改变任何参数。

第二步量化。rknn-toolkit2 1.6.0（WSL env `rknn16`），`target_platform='rk3588'`，`mean_values=[[0,0,0]]`、`std_values=[[255,255,255]]`，校准集 300 张（4 个视频各 75 张，`D:\rknn_calib`）。

第三步上板验证。先 `./rknn_probe models/xxx.rknn data/videos/test1.mp4 200` 看张量形状、`scale`、`zp` 和实跑值，再用 `./traffic_count test1` 跑完整视频，跟 FP32 结果逐框比。

这里最关键的一点：**必须让 DFL 距离单独成为模型输出**。沿用原模型的坐标输出，RKNN 会把 0～15 的距离和 0～640 的坐标塞进同一套量化参数，刻度被撑到 2.5，距离只剩 6 个有效台阶，框整体向左上偏，越远越严重。改造后距离的刻度是 0.047，中心偏移从 6.75 px 降到 0.03 px。完整推导见 [docs/head4_原理详解.md](docs/head4_原理详解.md)。

转换脚本在 WSL 仓库 `~/traffic_count/tools/`，不在板子上

## 八、关键技术难点与解决方案

### INT8 量化导致检测框整体向左上偏移

板端的框跟 PC 端 FP32 的结果对不上，整体往左上偏，车越小偏得越多。实测中心偏移 6.75 px：box1 左 6.4、上 14.5 px，box3 左 10、上 20 px。

原因是量化刻度的分配方式。RKNN 给每个张量按 `范围 / 256` 算刻度，原模型输出的是坐标（0～640），刻度被撑到 2.5017；而 DFL 内部的距离值只有 0～15，在这么粗的刻度下只剩 6 个有效台阶，精度基本被磨光了。

误差不是一步产生的，是六步逐级放大 53 倍：

```
DFL 距离 0~11.97        刻度 0.047
anchor − 左距离          刻度 0.325   ← 关键：大数(0.5~79.5)减小数(0~15)，小数值精度被稀释
anchor + 右距离          刻度 0.327
两者相加 ÷ 2             刻度 0.625
两者相减 = 宽高          刻度 0.084
拼 4 通道 × stride       刻度 2.502
```

改法是把模型输出换成 DFL 距离本身，坐标换算挪到 CPU 用 float 做：

- 从 ONNX 图里删掉 DFL 之后那 9 个换算节点（Slice×2 / Sub×2 / Add×2 / Div / Concat / Mul），把 `/model.22/dfl/Reshape_1_output_0` 设为图输出。这 9 个节点不含任何权重。
- 距离单独成一路输出后，RKNN 按它的真实值域分配刻度 `0.046943`，精度提升 53 倍。
- C++ 侧换算：`x1 = (anchor_x − 左距离) × stride`，`y1 = (anchor_y − 上距离) × stride`，右边界与下边界同理，改成加号。

结果：中心偏移 6.75 px → 0.03 px。模型权重没改，量化方式没改，推理速度也没变。

验证分三道关。ONNX 输出跟原模型逐元素差值为 0；用 numpy 重写一遍换算公式，跟原模型 boxes 差值为 0；板端实测 10 帧，左右上下偏移都不超过 0.04 px。

