# 项目交接文档（车流量统计 · RK3588 板子端）

> 最后更新：**2026-11-07**。**先看完这一份再动手。**
> **框偏移已解决**，原理详见 `docs/head4_原理详解.md`（带图，从零讲起）。

---

## 〇、30 秒速览（赶时间只看这段）

**项目**：YOLOv8 车流量检测跑在 Orange Pi 5 Plus（RK3588）NPU 上，带网页实时看。

**板子**：`ssh orangepi@192.168.1.105`（静态 IP，密码 orangepi），代码在 `~/code`。

**现在什么能用**：推理 + 跟踪 + 计数 + 网页，18.3 fps，跑 `models/best_detrac.rknn`（= **head4 新模型**）。

**框偏移：已解决。** 中心偏移从 6.75px 降到 **0.03px**（改善约 200 倍），速度不变。
做法：把 ONNX 的输出从「坐标 0~640」改成「DFL 距离 0~15」，坐标换算搬到 C++。
**模型权重一个字节没改，量化方式也没改。** 原理详见 `docs/head4_原理详解.md`。

**计数漏：已收尾。** 实测断号减少 34.3%（2559 → 1680 个编号），
计数净增 12，无一条线下降。剩余漏计成因未查证，投入产出比不高，暂不深究。
详见 6.1 节。

**已经堵死的路（别再试）**：
- rknn-toolkit2 的**手动混合量化**：1.6.0 崩 `KeyError: Concat_1_output_0`，2.5.0 转出来是坏的（详见 5.7）
- **2.5.0 转普通 INT8 也是坏的**（w/h 通道被撑大十几倍，见 6.2）
- `mmse` / `kl_divergence` / `hybrid_level` 这些量化参数：1 像素级收益，没用
- `count_dir` 只数一个方向 —— **不是缺陷**，这几个路口是单行道，设计如此

---

## 〇点五、文档地图

板子 `~/code/docs/` 是主力（板子是主力机器），仓库里的同名文件是同一份。

| 文档 | 内容 | 什么时候看 |
|---|---|---|
| `docs/MEMORY.md`（本文） | 交接文档：状态、连接、编译运行、踩坑、性能实测、待办 | **接手第一份** |
| `README.md`（板子根目录） | 项目简介、架构分层、目录结构、编译部署 | 想跑起来 |
| `docs/head4_原理详解.md` | 框偏移从零推导，11 节 11 图 | **想懂量化原理（本方向核心）** |
| `docs/技术文档.md` | 完整技术细节：算法、实测原始数据、方案对照 | 要查具体数字 |
| `docs/代码走读.md` | 逐文件逐函数讲解 | 要读代码 |
| `docs/实操手册.md` | 训练数据准备、参数标定、量化实验 | 要重训或重转模型 |
| `docs/INT8验证方案.md` | INT8 掉了多少精度（含实测结论） | 评估量化代价 |
| `docs/设计要点.md` | 待做改进清单（已标完成情况） | 规划下一步 |
| `docs/tools脚本清单.md` | 转换/诊断脚本怎么用 | 要转模型 |

**只想看性能数字**：直接跳到 **六点五节**（吞吐、延迟、实时流模拟、硬件上限）。

---

## 一、项目目标

把 YOLOv8 车流量检测跑在 Orange Pi 5 Plus（RK3588）的 NPU 上：

1. NPU 实时推理（目标 20fps+）
2. 跟踪 + 过线计数
3. 网页实时看画面和计数（板子没显示器，只能靠网页）
4. 最终用途：竞赛/论文的端侧部署 demo

---

## 二、连板子

```bash
ssh orangepi@192.168.1.105     # 密码 orangepi，sudo 也是 orangepi
```

| 项 | 值 |
|---|---|
| IP | **192.168.1.105（已设成静态）** |
| 网卡 | Realtek 无线（一块芯片，两个接口） |
| 系统 | Ubuntu 22.04.5，内核 5.10.160-rockchip |
| NPU 运行库 | `/usr/lib/librknnrt.so` **2.5.0**（别升级） |
| NPU 驱动 | 0.9.6（官方镜像自带，别动） |

**静态 IP 是怎么设的**（2026-11-01 做的，记下来免得再折腾）：

- NetworkManager 连接名 `700`
- `nmcli con mod 700 ipv4.method manual ipv4.addresses 192.168.1.105/24 ipv4.gateway 192.168.1.1 ipv4.dns 192.168.1.1`
- 已经重启验证过，MAC 怎么变都是 `.105`

**为什么必须用静态 IP**：这块卡会同时暴露 `wlan0` / `wlan1` / `wlP2p33s0` 三个接口，**接口名和 MAC（d0:39:57:cc:9b:eb 与 d2:...）每次开机可能互换**，靠路由器绑 MAC 根本治不住。

**板子现在没有外网**：路由器里留了两条 IP-MAC 绑定（`.105`/`.107`），板子的 MAC 一换就不匹配，路由器会把外网挡掉（SSH 和局域网不受影响）。要外网就让用户在路由器里删掉那两条绑定。

**WSL**：`wsl.exe -e bash -lc "<命令>"` 可用（早期版本记的「黑名单」已不成立）。
WSL 里 Python 环境是 `~/miniforge3`（不是 miniconda），
`conda activate rknn16` 是 rknn-toolkit2 1.6.0。

---

## 三、项目在哪

| 位置 | 路径 | 内容 |
|---|---|---|
| 板子 | `/home/orangepi/code` | C++ 主项目（编译、运行、网页都在这里） |
| WSL | `/home/imrong/traffic_count` | PC 端版本 + **模型转换脚本** |
| Windows | `D:\multi-road-crossing-vehicle-count` | 原始项目、视频、配置、ONNX |
| 校准图 | `D:\rknn_calib` | 300 张（4 个视频各 75 张） |

板子关键文件：

```
/home/orangepi/code/
├── README.md               # 项目说明（十节规范版）
├── traffic_count           # 命令行版（跑视频出结果）
├── traffic_web_single      # 单路网页版
├── traffic_web_quad        # 四路网页版（推荐）
├── rknn_probe              # 模型体检工具
├── multi_eval              # 多路并行压测
├── pipeline_bench          # 流水线压测
├── configs/test1~4.json    # ROI / 计数线 / 阈值
├── models/best_detrac.rknn # 模型（只有一个，见第五节）
├── data/videos/            # test1~4.mp4
├── output/                 # 结果视频、CSV、抓拍
├── docs/                   # 全部文档（见〇点五文档地图）
└── src/                    # 22 个 cpp/hpp，C++ 源码
```

**板端代码的正本在 WSL 仓库** `~/traffic_count`（git 仓库，远程
`multi-road-crossing-vehicle-count-cpp`）。`~/code` 不是 git 仓库，
代码改了要手动同步回 WSL 仓库再提交。

---

## 四、编译和运行

### 4.1 编译

```bash
cd ~/code
make -j4                    # 全部目标
make traffic_web_quad       # 只编四路网页版
make traffic_web_single     # 只编单路网页版
```

`make help` 能看到全部目标：`traffic_count`（命令行）、`traffic_web_single`、
`traffic_web_quad`（四路）、`rknn_probe`（模型体检）、`multi_eval`（多路压测）、
`pipeline_bench`（流水线压测）。

改了 `CMakeLists.txt` 的路径后**必须**重建缓存：

```bash
rm -f CMakeCache.txt && cmake . -DUSE_RKNN=ON -DRKNN_ROOT=/usr \
    -DTRAFFIC_PROJECT_ROOT=/home/orangepi/code && make -j4
```

### 4.2 运行

**命令行版**（跑完退出，结果在 `output/`）：

```bash
cd ~/code && ./traffic_count test1
```

**单路网页版**：

```bash
cd ~/code && ./traffic_web_single 8080
# 浏览器开 http://192.168.1.105:8080
```

**四路网页版**（推荐，开机就跑、视频循环）：

```bash
cd ~/code && setsid ./traffic_web_quad 8081 test1 test2 test3 test4 20 \
    > /tmp/quad.log 2>&1 < /dev/null &
# 浏览器开 http://192.168.1.105:8081
```

参数顺序是 `[端口] 配置1~4 [每路fps]`，都可省。**必须用 `setsid`**，
`nohup ... &` 在 SSH 非交互 shell 里会被回收。

### 4.3 停止

```bash
pkill -f "[t]raffic_web_quad"
pkill -f "[t]raffic_web_single"
```

方括号是为了不让 `pkill` 匹配到执行这条命令的 shell 自己。

### 4.4 几条注意

- `output: "show"` 在板子上**必崩**（没显示器），只能用 `"save"` 或 `"none"`
- 启动打印「服务已启动」**不代表成功**：端口被占时 `web_server.cpp` 会报错，
  但主程序还会再打印一遍（遗留代码 bug，别被忽悠）
- 网页上改的 conf/iou 只存内存，不写 json；进程重开回到配置文件的值
- 抓拍图存 `output/snapshots/`，**每次启动会清空**（`clear_snap_dir()`）

---

## 五、★模型转换（转不出来先看这一节）

### 5.1 板子上现有的模型

`models/` 里只有一个：

| 文件 | 大小 | 说明 |
|---|---|---|
| `best_detrac.rknn` | 5,969,848 | **head4 + 1.6.0 + INT8**，md5 `af1bfa9f11192d15fb30286067568651` |

这就是当前在用的模型，`TRAFFIC_PROJECT_ROOT "/models/best_detrac.rknn"` 写死在
`src/web_main_quad.cpp:73` 与 `src/web_main_single.cpp:70`。

> **2026-11-07 清理**：删掉 `v12_head4`（与主力 md5 完全相同，纯重复）、`v7`（旧 head3，
> 框偏 6.75px）、`fp16`（6.6 fps 太慢）、`old_ok`（10-30 旧版）。`models/` 34M → 5.7M。
> 之前 11-03 还删过 `v6` / `v8_h250` / `v9_h250` / `v10_int8` / `v11_auto` / `int8_normal`。
> **每个失败模型的原因都记在 5.7 与 6.2，不需要留文件。**

要用别的模型对比，用 `multi_eval --model <路径>` 临时指定，不要往 `models/` 里再堆。

### 5.2 转换配方（head3 的 v7 已验证通过，head4 在此基础上再改）

| 项 | 值 |
|---|---|
| 工具 | **rknn-toolkit2 1.6.0+81f21f4d**（WSL，`conda activate rknn16`，Python 3.10） |
| 输入 ONNX | `best_detrac_head3.onnx`（12.77 MB，**删掉了末端 Concat**） |
| ONNX 输出名 | `/model.22/Mul_2_output_0`（boxes [1,4,8400]）+ `/model.22/Sigmoid_output_0`（scores [1,1,8400]） |
| config 参数 | `target_platform='rk3588'`、`mean_values=[[0,0,0]]`、`std_values=[[255,255,255]]` |
| build 参数 | `do_quantization=True`，其余全默认（w8a8 / normal / optimization_level 默认 1） |
| 校准集 | `D:\rknn_calib` 300 张（`dataset_calib_head3.txt` 清单） |
| 产出 | `~/traffic_count/models/best_detrac_v7.rknn`，**6,482,288 字节** |
| 验证结果 | 20.6 fps、4.1 框/帧、test1 前 400 帧 A 线 4 辆、box0 `x1=161.8 y1=-2.7` |

### 5.3 最关键的一条：ONNX 末端绝对不能有 Split

**失败现象**：`want_float=1` 时分数全 0。

**原因**：之前把「拆双输出」做成了「在末端 Concat 后面加一个 `Split(axis=1,[4,1])`」。RKNN 编译器把输出上的 Split 融合掉，两个版本各有各的错法：

| ONNX | toolkit | 结果 |
|---|---|---|
| Concat 保留 + 末尾 Split（head2） | 1.6.0 | scores 的 8400 个 int8 **全是 -128**（= zp = 0.0） |
| 同上 | 2.5.0 | scores 正常，但 boxes 塌（float 均值 330 vs 正常的 182），框挤在左边缘 |
| **删掉 Concat，两个张量直接当输出**（head3） | 1.6.0 | ✅ 正常 |

**正确做法**（`tools/01_convert/make_head3.py` 干的事）：

```python
# 删掉末端 Concat，把它的两个输入直接当 graph output
m.graph.node.remove(concat)          # concat.name = '/model.22/Concat_3'
del m.graph.output[:]
m.graph.output.extend([
    helper.make_tensor_value_info(boxes,  TensorProto.FLOAT, [1, 4, 8400]),
    helper.make_tensor_value_info(scores, TensorProto.FLOAT, [1, 1, 8400]),
])
```

### 5.4 ★head4：框偏移的最终解法（2026-11-02 验证通过）

> **要完整搞懂这套方法，看 `docs/head4_原理详解.md`（11 节、11 张图，从「什么是量化刻度」讲起）。**
> 本节只记结论和参数，方便写代码时查。

**一句话**：把模型输出从「坐标 0~640」换成「DFL 距离 0~15」，坐标换算搬到 C++。

**为什么有效**：RKNN 给输出配刻度是 `范围/256`。原来输出是坐标（0~640）→ 刻度 2.5，
但 DFL 内部的距离只有 0~15，被这把粗尺子一压，精度全丢。
距离单独当输出后，刻度变成 `0.047`，**细了 53 倍**。

**更完整的原因**（2026-11-02 补充，读 `quantization.cfg` 实测）：
不只是「最后配了个粗刻度」，而是**中间那几步每一步都在 int8 里重新量化**。
从 DFL 到输出一共 6 步，刻度一路被撑大：

| 步骤 | 值域 | 刻度 |
|---|---|---|
| DFL 输出距离 | 0 ~ 11.97 | **0.047** |
| anchor − 左距离 | −3.47 ~ 79.46 | 0.325 |
| anchor + 右距离 | 0 ~ 83.28 | 0.327 |
| 两者相加 ÷2 | 0 ~ 159.48 | 0.625 |
| 两者相减 = 宽高 | 0 ~ 21.43 | 0.084 |
| 拼 4 通道 × stride | 0 ~ 637.92 | **2.502** |

关键在**第 2 步**：`anchor`（0.5~79.5 的大数）减去 `距离`（0~12 的小数），
结果范围被大数主导，小数的精度立刻被稀释约 7 倍。后面又乘又加，一路涨到 2.5。

**所以 head4 修的是「后面 5 次量化全都不发生了」**，不只是换个刻度。
C++ 里用 float32（24 位有效数字），根本没有「刻度」这回事。

**实测结果**（10 帧，每帧和 FP32 真值逐框对比）：

| | 左边缘 | 上边缘 | 中心偏移 |
|---|---|---|---|
| v7（旧） | −4.94 | −5.50 | **6.75 px** |
| head4（新） | +0.04 | −0.00 | **0.03 px** |

速度不变（18.3 fps），因为全是 int8，没有 fp16 层。

**改了什么**（`tools/01_convert/make_head4.py`）：
1. 删掉 DFL 之后的 9 个节点（Slice/Sub/Add/Div/Concat/Mul ×2）——**这些节点没有任何权重**
2. 把 `/model.22/dfl/Reshape_1_output_0` 直接设成图输出

**C++ 侧换算**（`rknn_detector.cpp`）：

```cpp
x1 = (anchor_x - 左距离) * stride;
y1 = (anchor_y - 上距离) * stride;
x2 = (anchor_x + 右距离) * stride;
y2 = (anchor_y + 下距离) * stride;
```

| 元素 | 规格 |
|---|---|
| anchor 网格 | 三段：80×80、40×40、20×20，值 = (列号+0.5, 行号+0.5) |
| stride | **是 8400 长的数组，不是标量！** 前 6400 个=8，中 1600 个=16，后 400 个=32 |
| 距离顺序 | 左、上、右、下 |

**自动识别**：`rknn_detector.cpp` 看坐标输出张量名里有没有 `dfl`
（head4 是 `/model.22/dfl/Reshape_1_output_0`，head3 是 `/model.22/Mul_2_output_0`）。
**所以 v7 和 head4 可以随便换，代码都能跑。**

**三道验证关**（改模型必过）：
1. ONNX 合法 + **分数输出和原模型差值必须为 0**
2. numpy 重写换算公式，和原模型 boxes 差值必须为 0（实测精确 0）
3. 板子上和 FP32 真值逐框对比

**关于 `head4` 这个名字**

`head` 指模型的**检测头**（head），即负责输出结果的最后一段网络。
YOLOv8 的检测头在 ONNX 图里节点名前缀是 `/model.22/`，官方叫 `Detect` head。

本项目为了绕开 RKNN 的量化问题，对导出结构改了三版，用编号区分：

| 名称 | 输出结构 | 改动 | 状态 |
|---|---|---|---|
| 原始 | 单输出 `[1,5,8400]` | 无（ultralytics 导出原样） | 分数全 0，不可用 |
| head2 | 双输出 | 末端加 `Split` 拆分 | 触发工具 bug，不可用 |
| head3 | 双输出 `[1,4,8400]` + `[1,1,8400]` | 删掉末端 `Concat` | 可用，但框偏 6.75 px |
| **head4** | 同上形状，内容改为距离 | 再删 DFL 之后 9 个换算节点 | **当前主力，框偏 0.03 px** |

**这三个编号（head2/3/4）是本项目自定的，不是 YOLO 或 RKNN 的官方术语。**
官方只有 `Detect` head 一个概念，没有 head3、head4 之分。

换个说法也成立：head4 就是「输出 DFL 距离而不是坐标」的那版模型。
如果名字让人困惑，直接按特性称呼即可。

**三种模型的输出形状**（容易混，记一下）：

| | 输出 | 那 4 行代表 | 值域 |
|---|---|---|---|
| 原始 ONNX | `[1,5,8400]` 单个 | cx cy w h + 分数 | 0~640 |
| head3 | `[1,4,8400]` + `[1,1,8400]` | cx cy w h | 0~640 |
| head4 | `[1,4,8400]` + `[1,1,8400]` | 左 上 右 下 距离 | 0~15 |

**head3 和 head4 形状完全一样，只有含义不同**，所以后处理必须靠张量名判断。

**为什么可以随便改输出**：ONNX 规范里 `graph.output` 就是一行声明，
任何张量都能声明成输出。`onnx.checker` 通过、onnxruntime 正常跑、
权重一个字节没改、换算公式验证差值为 0。
**代价**是后面那几步得自己写对（anchor 网格、stride 分段、距离顺序，错一个全错）。

### 5.5 怎么复现

**head4（当前主线）：**

```bash
# Windows 侧
cd D:\multi-road-crossing-vehicle-count
D:\Python3.12.4\python.exe tools\01_convert\make_head3.py
D:\Python3.12.4\python.exe tools\01_convert\make_head4.py
D:\Python3.12.4\python.exe tools\01_convert\verify_head4_decode.py   # 必须看到差值 0

# WSL 侧
conda activate rknn16
cd ~/traffic_count
python /mnt/d/multi-road-crossing-vehicle-count/tools/01_convert/convert_head4_probe.py
```

**head3（老路线，对照用）：**

```bash
conda activate rknn16
python /mnt/d/multi-road-crossing-vehicle-count/tools/03_failed/convert_head3.py
```

### 5.6 怎么验证（两步，别只看帧率）
**第一步：模型体检**（工具已经在板子项目里：`src/rknn_probe.cpp`，编出来是 `~/code/rknn_probe`）

```bash
cd ~/code
make rknn_probe        # 第一次或改了源码才编，之后直接用
./rknn_probe models/新模型.rknn data/videos/test1.mp4 200
```

它干的事：打印每个输入/输出张量的形状、类型、scale、零点，然后真跑一帧，把输出**原样摊开**（反量化后的 float + 未换算的 int8）。换新模型先跑它，别拿业务程序猜。

合格标准（第 200 帧，test1）：

| 输出 | 合格表现 |
|---|---|
| boxes float | min 0、max ≈630、**均值 ≈182** |
| scores int8 原始值 | 8400 个里约 **60 个非零、9 个 >0.5**（8000 多个 -128 是正常的，大多数 anchor 没车） |
| scores float | max ≈0.78~0.80 |

**第二步：跑 400 帧看真实结果**

```bash
sed -e 's/"name": "test1"/"name": "tmpv7"/' -e 's/"max_frames": 0/"max_frames": 400/' \
    configs/test1.json > /tmp/t1v7.json
# 换模型前先备份 models/best_detrac.rknn，跑完必须还原
cd /tmp && /home/orangepi/code/traffic_count /tmp/t1v7.json
```

合格：`box0` 的 `x1` 在 150~170、`y1` 是 -3 左右、约 4 框/帧、A 线 4 辆。

### 5.7 混合量化：试过，走不通（别再试）

**目的**：INT8 的框偏移想让 DFL 那几层走 fp16、其余仍 int8，既保精度又不掉速度。

**结论：1.6.0 和 2.5.0 都走不通，别再花时间。**

| 版本 | 结果 |
|---|---|
| 1.6.0 | step2 崩在 `quant_optimizer._p_adjust_tanh_sigmoid`（`KeyError: '/model.22/Concat_1_output_0'`）。改 35 层、改 2 层、只改 dtype 三种配置崩在同一行 → 工具自身 bug，绕不过去 |
| 2.5.0 | step1 能过（必须 `proposal=False`），step2 能出模型，但**转出来是坏的** |

2.5.0 出的两个模型都坏：

- **v8_h250**（DFL 和 Mul 都 fp16）：输出变 FLOAT16 后编译器插了 NC1HWC2 布局重排，
  元素顺序不再是 NCHW。anchor 0 的 `cy` 应该是 5，读出来是 397。
- **v9_h250**（只有 DFL 内部 fp16，输出仍 int8）：DFL 距离 0~15 和下游坐标 0~640
  共用一套量化参数，0~15 被压成 6 个台阶，4 个通道范围全变成 0~638。

**这不是混合量化功能的问题，是量化参数共用的问题**（根因见 6.2）。
调量化救不了，正确解法是 head4。这两个模型已从板子删除，失败原因记在这里就够了。

### 5.8 试过但没用的（别再重复）

- `quantized_dtype`、`hybrid_level`、`custom_string`、`quantized_input_type` —— 全是猜的，无效
- `mmse` / `kl_divergence` —— 只是换裁剪范围，1 像素级收益；mmse 还容易 OOM
- 2.5.0 的 `disable_rules=['seperate_conv_rs_split']` —— 只对 2.5.0，1.6.0 没这个参数
- `optimization_level=3` —— 会把 DFL 坐标算飞（框巨大错位），只能用默认/1
- **rknn-toolkit2 的混合量化**（详见 5.7）—— 两个版本都有 bug

**结论：模型这块停在 1.6.0 + head4 + INT8（`best_detrac.rknn`）。这是当前最优，不要再折腾。**

---

## 六、已知问题

### 6.1 计数漏（2026-11-02 收尾）

**现状**：问题已处理告一段落，不再继续追查。断号这一支已证实并修复，
剩余漏计不影响交付，记录如下备查。

#### 原来的三个可疑点，逐一交代

| # | 可疑点 | 结论 |
|---|---|---|
| 1 | `counter.cpp` 的 `near_segment` 不通过就 `continue`，连 `prev_` 状态都不更新 | **未查证**。理论上转弯车轨迹偏出线段端点会判不到，但这只是推测，没有实测数据支撑 |
| 2 | `count_dir: "neg"` 只数一个方向 | ❌ **不是问题**。这几个路口都是**单行道**，单方向计数是设计意图。除非出现逆行车辆，否则不会有漏计 |
| 3 | 跟踪断号：转弯时 IoU 匹配容易掉，新 ID 没有「上次在哪一侧」的记录 | ✅ **已证实并修复**，见下 |

> **第 2 条不要再当成缺陷去查。** 单行道场景下它是对的，
> 改动它反而会把逆行误判成正常车流。

#### 实测：框偏移修复后，断号减少 34.3%

2026-11-02 用 `multi_eval`（板端多路并行评测）跑完整视频，两模型同配置对比：

| 视频 | 帧数 | v7 编号数 | head4 编号数 | 降幅 |
|---|---|---|---|---|
| test1 | 3600 | 553 | **450** | 18.6% |
| test2 | 3600 | 625 | **441** | 29.4% |
| test3 | 2535 | 909 | **539** | 40.7% |
| test4 | 2400 | 472 | **250** | 47.0% |
| **合计** | 12135 | **2559** | **1680** | **34.3%** |

（编号数 = `Tracker::total_ids()`，累计分配过多少编号。同一视频同样帧数下，
编号越少说明断号越少。）

计数变化：净增 **+12**（test1 A +2、test3 D +8、test4 A/B 各 +1），**没有一条线下降**。

#### 结论

**断号减少 34.3%，四个视频无一例外，越密的场景降得越多。**
这支持因果链：**框偏移 → 框抖 → IoU 匹配失败 → 断号 → 漏计**。

但计数只多 12，且 test2 的 A/B 两条线完全没变。
说明框偏移只解释了约三分之一的漏计。

**剩下的漏计暂不深究** —— 视频里过车基数本来就小（5 分钟几十辆），
计数差异的统计意义有限；再看下去投入产出比不高。

#### 复现方法

```bash
# 板子上，四路并行，各跑 3600 帧（test3/test4 视频本身更短，跑完即止）
cd ~/code
./multi_eval configs/test1.json configs/test2.json configs/test3.json configs/test4.json \
    --frames 3600 --out output/full_head4
./multi_eval configs/test1.json configs/test2.json configs/test3.json configs/test4.json \
    --frames 3600 --model /path/to/other.rknn --out output/full_other

# 把两个 summary.csv 拉回 PC 后对比
python tools/02_diagnose/compare_models.py full_v7_summary.csv full_head4_summary.csv
```

实测速度：4 路并行合计 **62~73 fps**（单路 18~22 fps），12135 帧约 3 分钟跑完。

### 6.2 框偏移（2026-11-02 **已解决**）

**现在的状态**：中心偏移 6.75px → **0.03px**，速度不变。解法见 **5.4 节**，
完整原理（带图）见 `docs/head4_原理详解.md`。

下面是当时的诊断过程，留着备查。

**现象**：板端框整体向左上偏，**车越小（越远）偏得越厉害**。截图见 `docs/screenshots/int8_shift_*.png`。

**真正的根因**（2026-11-02 查明，此前一直猜错）：
RKNN 给模型输出配量化刻度用的是 `刻度 = 输出范围 / 256`。
原来模型输出的是**坐标（0~640）**，所以刻度 = 640/256 = **2.5**。
但 DFL 内部那个「距离」只有 0~15，被这把粗尺子压得只剩几个台阶 → 精度全丢。

**决定性证据**（同一帧、同一个车，anchor #2180）：

| 模型 | cx | cy | **w** | **h** |
|---|---|---|---|---|
| v7（正确） | 160.1 | 215.1 | **20.0** | **20.0** |
| v10（2.5.0 坏的） | 120.1 | 195.1 | **235.2** | **260.2** |

w/h 被撑大 **13 倍**——所以「框偏左上」其实是「框被撑大后中心偏移」。
另外 v10 里 w/h 通道的最大值正好是 **637.92 = 255 × 2.5**（int8 的饱和值），
说明距离值超出了刻度能表示的范围，被截断了。

**4 个通道范围对比**（好模型四路各不相同，坏模型四路一样）：

```
v7  : ch0 0~630 | ch1 0~615 | ch2 2.5~510 | ch3 2.5~230   ← 正常
v10 : ch0 0~638 | ch1 0~638 | ch2 10~638  | ch3 10~638    ← 坏了
v11 : 四路全是 0~638                                        ← 坏得更彻底
```

**当时猜错的地方**（教训）：
一直以为是「DFL 的 16 个 bin 精度不够」，于是去折腾混合量化。
实际是**刻度配错了范围**——不是精度不够，是精度被浪费了。
方向错了，所以怎么调都没用。

**已排除的怀疑**（都验证过）：
- 不是 letterbox 补边算错（ONNX 侧同样算法结果正常）
- 不是反量化公式错（`(q-zp)*scale`，zp=-128 验证过）
- 不是 `want_float` 的问题（探针实测反量化后的值是对的）
- 不是模型权重问题（旧模型、v7 模型偏移量一致）

**其他方案的现状**：

| 方案 | 精度 | 速度 | 结论 |
|---|---|---|---|
| **head4（现在用的）** | **偏 0.03px** | 18.3 fps | **✅ 已采用** |
| v7（旧） | 偏 6.75px | 18.3 fps | 留作对照 |
| 纯 FP16 | 与 FP32 无差 | 6.6 fps | 太慢，没必要了 |
| 混合量化 | — | — | 两个 toolkit 版本都失败（5.7） |
| 后处理补偿 | 治标 | 不变 | 不需要了 |


### 6.3 框抖

已删掉重复的 EMA（见下），只保留卡尔曼一层平滑。

2026-11-01 改动：**删掉了 tracker.cpp 里的 EMA 平滑**（`0.6×旧 + 0.4×新`）。它和卡尔曼重复平滑——卡尔曼那层滤波结果才是真正该用的，输出框现在直接取 `kf.state()`。实测计数和框位置完全不变（400 帧 A=6 前后一致），说明 EMA 只是白加 1~2 帧滞后。

### 6.4 其他

- 网页端口被占时仍打印「服务已启动」：`main.cpp` 无条件打印

### 6.5 已修：prev_ 只增不删

`Counter::prev_` 原来只增不删，接摄像头连跑几天会攒到几十万条。2026-11-01 加了 `retain(live_ids)`：每帧问跟踪器"现在还有谁活着"，不在名单上的清掉。跟踪器删 id 的时机是它自己定的（`track_buffer_=30` 帧），计数器不自己数帧，两边不会错位。

---

## 六点五、性能实测数据（2026-11-02）

三组测试，覆盖三种用法：批量多视频、单视频流水线、模拟实时流。

### ① 批量多视频并行（`multi_eval`）

每路一个线程，按 `i % 3` 绑 NPU 核。不画图不存视频。

| 线路数 | 合计吞吐 | 相对单路 |
|---|---|---|
| 1 | 17.3 fps | 1.00× |
| 2 | 37.9 fps | 2.19× |
| 3 | 58.3 fps | 3.37× |
| 4 | 71.2 fps | 4.12× |

**线性扩展。** 12135 帧四路并行约 3 分钟跑完（单路串行要 11 分钟）。

### ② 单视频流水线（`pipeline_bench`）

三段流水：读帧线程 → 推理线程（K 个，各绑一核）→ 跟踪线程（单线程串行）。
帧号作钥匙，按序重组后再送跟踪。

| 推理线程数 | 吞吐 | 相对单线程 |
|---|---|---|
| 1 | 18.4 fps | 1.00× |
| 2 | 41.7 fps | 2.27× |
| 3 | 65.9 fps | 3.58× |
| 4 | 85.0 fps | 4.62× |
| 6 | 105.5 fps | 5.73× |
| 8 | 113.2 fps | 6.15× |
| 10 | **118~121 fps** | **6.6×** |
| 12 | 117.0 fps | 6.36× |
| 16 | 120.1 fps | 6.53× |

**拐点在 10 线程。** 超过 10 之后不再上升，因为 3 个核已经吃满。

### ③ 跟踪到底占多少

| 线程数 | 只检测 | 检测+跟踪 | 差异 |
|---|---|---|---|
| 3 | 70.7 fps | 66.7 fps | -5.7% |
| 6 | 107.3 fps | 107.6 fps | 0 |
| 10 | 117.4 fps | 116.9 fps | 0 |

**跟踪几乎不占时间。** 因为跟踪约 2~3ms/帧，只有 NPU 的 5%，
而且跑在独立线程里，被 NPU 的时间完全掩盖。

**结论：不需要把跟踪挪到后面批量做。「边跑边追」没有额外成本。**

### ④ 延迟拆解（为什么大数字会骗人）

| 线程数 | 纯推理 | 排队等待 | 端到端 |
|---|---|---|---|
| 1 | 52.5 ms | 1670 ms | 1725 ms |
| 3 | 49.8 ms | 501 ms | 553 ms |
| 10 | 118.7 ms | 296 ms | 418 ms |

**端到端那 1.7 秒里，1.67 秒是在队列里排队，真正算一帧只要 52ms。**

排队时间 = 队列长度 × 单帧耗时（32 帧 × 52ms ≈ 1.66 秒，与实测吻合）。

### ⑤ MP4 模拟实时流（`--realtime N`）

让读帧线程按指定 fps 节奏送帧，等价于摄像头推流。
判据看**队列占用率**，不看吞吐（送帧速度受节奏限制，吞吐必然约等于送帧速度）。

| 送帧速度 | 1 线程 | 3 线程 | 6 线程 | 8 线程 |
|---|---|---|---|---|
| 15 fps | ✅ 65ms | ✅ 62ms | — | — |
| 30 fps | ❌ 队列 100% | ✅ **47ms** | — | — |
| 45 fps | ❌ | ✅ 47ms | — | — |
| 60 fps | ❌ | ✅ 47ms（队列10%） | — | — |
| 90 fps | ❌ | ❌ | ✅ 62ms | ✅ 68ms |
| 120 fps | ❌ | ❌ | — | ✅ 146ms |

**关键结论：**

- **单线程只能扛 15fps 的流**（单核推理 20fps，30fps 就积压）
- **3 线程能扛 60fps 的流，延迟仅 47ms**
- 8 线程能扛 120fps，但延迟涨到 146ms

**所以接摄像头确实必须开多线程。** 2 线程就够 30fps，3 线程有余量。

### ⑥ 实测的硬件上限

```
单帧推理（单核）:  42~52 ms  →  单核 20~24 fps
3 核理论极限:      3 × 24 = 72 fps（无竞争时）
流水线实测峰值:    120 fps（10 线程，靠队列深度掩盖延迟）
```

**120 fps 已接近这块板子的物理上限**，再堆线程无收益。

### ⑦ 用这些数据的场景建议

| 场景 | 配置 | 预期 |
|---|---|---|
| 批量跑多个 mp4 | `multi_eval`，4 路并行 | 71 fps，约 3 分钟/12135 帧 |
| 单个 mp4 要快 | `pipeline_bench --workers 10 --buf 64` | 118 fps |
| 接 30fps 摄像头 | `--workers 3 --buf 30 --realtime 30` | 47ms 延迟，跟得上 |
| 接 60fps 摄像头 | `--workers 3 --buf 30 --realtime 60` | 47ms，队列 10% |

> `--buf` 是队列上限。批处理时设大（64）提高吞吐；实时流设小（30 或更小）压低延迟。

### ⑧ GPU 情况

`/dev/mali*`、`g18p0` 驱动、`fb000000.gpu` 调频节点都在，是 **Mali-G610 MP4，最高 1.0 GHz**。

**但当前项目完全没用它。** 推理在 NPU，跟踪/计数/画图在 CPU。

值得用的只有一处：存视频时用硬件编码器（rkmpp），能省 CPU。但目前不做视频输出。

**不能用来训练**：Mali 没有可用的 PyTorch 后端，且 6GB 共享内存、25GB/s 带宽都不够。

---


## 七、踩的坑（都是真踩的）

只留以后还会再踩、且知道了能省事的。

1. **`output: "show"` 在板子上必崩**（没显示器），只能用 `"save"` 或 `"none"`。
2. **改了 `CMakeLists.txt` 的路径后不删 `CMakeCache.txt`** → 一直用旧路径，
   症状是编译过了但运行找不到模型/配置。必须 `rm -f CMakeCache.txt && cmake . -DUSE_RKNN=ON`。
3. **CPU 调频影响帧率**：默认 `ondemand` 实测 16.0 fps，设 `performance` 后 20.6 fps。
   开关在 `/sys/devices/system/cpu/cpufreq/policy{0,4,6}/scaling_governor`，重启失效。
   跑性能测试要切 `performance` 才公平。
4. **`pkill -f traffic_web_single` 会杀掉自己那条命令**（pattern 匹配到 `bash -c`），
   要用 `pkill -f "[t]raffic_web_single"`。`pkill -x` 也不行，进程名被截断成 15 字符。
5. **网页上改的 conf/iou 只存内存**，不写 json，进程重开就回到配置文件的値。
   四路版启动时从各自 `configs/testN.json` 读一次当初值（`WebServer::init_params()`）。
6. **板子重启后 `/tmp` 清空**，编译的探针、临时脚本都要重传重编。
7. **DFL 的 stride 是 8400 长的数组，不是标量**。前 6400 个是 8、中 1600 个是 16、
   后 400 个是 32。当成标量 8 算，后两段全错（实测最大差 453 像素）。
8. **anchor 网格也要分段生成**：80×80 / 40×40 / 20×20，值 = (列号+0.5, 行号+0.5)。
   直接线性排 0.5~79.5 是错的。
9. **`rknn_probe` 只打印前 12 个值**，看不出是哪个通道坏了。要判断通道语义得用
   `tools/02_diagnose/ch_scan.cpp`（每通道独立统计 min/max）。

---

## 八、待办（下一个 agent 从这里开始）

### 当前状态一句话

板子跑 `best_detrac.rknn = head4`（1.6.0 + head4 + INT8），18.3 fps，
**框偏移已解决（0.03px），计数漏已收尾（断号降 34.3%）**。
模型转换这条路已经走到头，不要再碰。

### 仓库提交状态（2026-11-06 晚更新）

| 仓库 | 提交 | 推送 |
|---|---|---|
| `D:\multi-road-crossing-vehicle-count`（Python + 文档） | `a4d4a25`「按工程规范重写 README」 | 已推送 |
| WSL `~/traffic_count`（C++ 板端） | `73c425d`「按工程规范重写 README」 | 已推送 |

两个仓库现在都跟 origin 同步，`git rev-list --left-right --count origin/main...HEAD` 都是 `0 0`。

**最近三个提交**：

- `a4d4a25` / `34cd9d7`「按工程规范重写 README」——新增 WSL 仓库 `docs/代码走读.md`（867 行）：
  逐文件讲编译期常量、三个路径函数、RKNN 后端的双输出拆分与 head4 换算、卡尔曼的两条教训、
  匈牙利算法的完美匹配陷阱、counter 的六步判过线、四路版 16 线程的分核与丢帧兜底；
  两边 README 的配套文档表各加一行指向它。
- `6deb6d3` / `a6a41c7`「按工程规范重写 README」——README 改为十节固定结构（项目简介 / 技术栈 /
  系统架构 / 环境依赖 / 项目结构 / 编译运行部署 / 核心功能模块 / 技术难点 / 版本记录 / 作者与声明），
  操作层长文归档到 D 仓库 `docs/实操手册.md`（172 行）与 WSL 仓库 `docs/技术文档.md`（813 行）。
- `9ffc855`「加四路分屏网页版」——四路网页版整套代码第一次入库：新增 `src/web_main_quad.cpp`
  （648 行）与 `web/quad.html`；`src/web_server.cpp` 507→580 行、`web_server.hpp` 146→172 行；
  `web.cmake` 加 `traffic_web_quad` 目标；`src/rknn_detector.cpp/hpp` 加 `set_core()` +
  `rknn_set_verbose()`；删 9 个弃用转换脚本。
- `01c71e7`「加四路分屏网页版」（D 仓库，仅改 MEMORY.md）。

> **推送踩坑（纠正此前的错误结论）**：本机 `git config --global http.proxy` 有值，为
> `http://127.0.0.1:7892`，对应代理软件的本地端口。**代理软件开着时，加
> `-c http.proxy= -c https.proxy=` 反而会失败**（直连 443 被阻，报
> `Failed to connect to github.com:443 after 21066 ms`），直接 `git push origin main` 就能通。
> 反之，代理软件没开时才需要用 `-c` 绕过（否则卡在 `after 0 ms`）。
> 判断方法：`netstat -ano | findstr LISTENING | findstr 7892`，有监听就直接 push。

> **踩坑（2026-11-06）**：D 仓库和 origin 曾各自有一个「修复框偏移」提交（本地
> `ca9fc45`、远端 `0faaa7f`），只差 MEMORY.md 里 8 行仓库状态表，导致 `ahead 1,
> behind 1`。处理办法是 `git reset --hard origin/main` 后把本地那几行重新 apply，
> 不要 force push 覆盖远端。

### 仓库结构清理（2026-11-06）

做了四件事，都是纯结构问题，不影响任何功能：

1. **删掉 `.git.old/`** —— 2026-10-18 首次提交时误把旧 git 目录整个提交进去了，
   173 个文件 / 18M。`git rm -r --cached .git.old` 后从磁盘删除。历史里仍有这个对象
   （要彻底清干净得 `git filter-repo`，但不影响 clone 速度太多，暂不做）。
2. **D 仓库不再存板端 C++ 镜像** —— 原先 `src/rknn_detector.cpp`、`web_main_quad.cpp`、
   `web_server.cpp`、`web_server.hpp`、`rknn_detector.hpp`、`web.cmake`、`web/quad.html`
   在 D 仓库是未跟踪副本，正本在 WSL 仓库。两份容易改漏一边，已删除 D 侧副本。
   以后板端代码**只改 WSL 仓库 `~/traffic_count`**，D 仓库只是 Python 版与文档。
   `docs/head4_原理详解.md` 与 `docs/实操手册.md` 里提到 `src/rknn_detector.cpp` 的地方
   指的是 WSL 仓库里的那份。
3. **加 `LICENSE`** —— 学术使用许可（Academic Use Only），两个仓库同一份（md5
   `d4c2d87e7923b3a15aeecd4f6f1fb459`，2092 字节）。README 第十节已改为链接到它。
4. **加 `.gitattributes`** —— 统一行尾规则：文本 `text=auto`（仓库内 LF），二进制
   （png/jpg/webp/mp4/onnx/rknn/pt/npy/zip）不做转换。起因是 WSL 与 Windows 两侧
   md5 对不上，实际只是 CRLF 差异。

另外 WSL 仓库执行过 `git config core.filemode false`：此前 `src/rknn_detector.cpp`
长期显示 `M`，`git diff` 只有 `old mode 100755 / new mode 100644`，内容零改动，
是文件系统权限位不稳导致的误报。

### tools 目录结构（2026-11-02 整理过）

```
tools/
├── 01_convert/     转换主线 —— 要转模型只看这里
├── 02_diagnose/    诊断验证 —— 判断模型好坏
├── 03_failed/      失败记录 —— 别再跑
├── 04_dataset/     数据集相关
└── 05_figures/     画讲解图
```

详细说明见 `tools/README.md`。

### 当前状态（2026-11-06）

框偏移已解决，计数漏已收尾，四路分屏网页版已入库并推送。
板子的四路参数默认值不再写死，改成启动时读 `configs/testN.json` 的 `detect.conf/iou`。
两个仓库的 README 已按企业级规范重写（十节固定结构），长文归档到 `docs/`。
仓库结构已清理：删 `.git.old`、去重复的 C++ 镜像、加 LICENSE 与 .gitattributes。

如果以后要继续提升，按投入产出比排序：

0. **四路分屏页面卡顿**（唯一新出现的问题）—— 4 条 MJPEG 常驻流单条约 500 KB/s，
   占满浏览器对同一主机的 6 个并发连接，`/api/state` 轮询只能排队。
   治法：缩略图降到 3~5 fps 或降分辨率，或把 5 次轮询合并成一次请求。

1. **`near_segment` 那条（未查证）** —— 唯一还没验证过的怀疑点。
   做法：给 `counter.cpp` 加轨迹日志（每帧每车：id、中心、在线的哪一侧、
   是否 near_segment），跑完整 test1 和 CSV 对账。
2. **扩大统计基数** —— 现在 5 分钟视频只有几十辆车，计数差异统计意义有限。
   跑完整 15 分钟视频（`--frames 0`）、或多跑几个场景，结论会更稳。
3. **板端多路实时跑、测功耗** —— 部署收尾工作，与算法无关。

> **不要做的事**：`count_dir` 只数一个方向不是缺陷（单行道场景），
> 模型转换这条路也走到头了（见速览的「已经堵死的路」）。


### 2026-11-07 做的事

1. **代码头注释**（只改板子，未推仓库）
   核心文件本来就有合格头部，本次只补 `src/tracker.cpp` 一段（11 行，讲五段流程 +
   两个关键点：低分框只救不建、输出不叠 EMA）。其余 9 个非核心文件各加 2~3 行，
   只讲"这是什么、干什么用"，不写实现细节：`web_main_single.cpp`、`web_main_quad.cpp`、
   `web_server.cpp`、`detector.cpp`、`mini_json.hpp`、`rknn_probe.cpp`、`multi_eval.cpp`、
   `pipeline_bench.cpp`、`recorder.hpp`。`make -j4` 六个目标全部通过。

2. **抓拍图改为启动时清空**
   原先每次过线抓一张、从不清理，实测堆了 **9134 张 / 100M**。
   `web_main_quad.cpp:77` 与 `web_main_single.cpp:76` 各加一个 `clear_snap_dir()`，
   在 `main()` 开头调用。启动打印「已清空抓拍目录, 删除 N 张旧图」。
   实测有效，`output/snapshots/` 现在启动即为空。

3. **模型目录清理**
   删掉 4 个没用的，只留 `best_detrac.rknn`（head4 主力）：
   `v12_head4`（与主力 md5 完全相同，纯重复）、`v7`（旧 head3，框偏 6.75px）、
   `fp16`（6.6 fps 太慢）、`old_ok`（10-30 的旧版）。`models/` 34M → 5.7M。

4. **docs 精简**
   删掉两个归档（`归档-板子端README-2026-11-02.md`、`归档-项目记忆-2026-10-22.md`）
   与 `文档索引.md`（内容已过时，且指向的其他文档都在）。详见下节文档地图。

5. **VS Code 远程每次都要下载 server 的根因**
   PC 端 VS Code 默认自动更新，版本号一变板子就要配套下载新 server；板子无外网，
   下载走 SSH 隧道所以慢。查证：`~/.vscode-server/cli/servers/` 有五个版本各 600M+，
   共 3.7G。删掉 4 个旧版 + 旧式 `bin/`，只留当前在用的
   `Stable-c4453f8ea430795f941c1b2ff6c9321135a74a02`，3.7G → 757M。
   **治本**：PC 端 `Ctrl+,` 搜 `update mode`，改成 `manual` 或 `none`，否则还会再长。


### 已验证可用的工具

**板子 `~/code` 下（永久）：**

| 工具 | 用途 |
|---|---|
| `./rknn_probe 模型 视频 帧号` | 模型体检：打印张量属性 + int8 原始值 + 反量化值 |
| `./traffic_count 配置` | 命令行跑视频（画图存视频，单路） |
| `./multi_eval 配置... --frames N` | **多路并行评测**，不画图不存视频，最快；支持 `--model` 换模型对比 |
| `./traffic_web_single 8080` | 网页实时看（`pkill -f "[t]raffic_web_single"` 关掉） |

**`multi_eval` 用法**（2026-11-02 新增）：

```bash
cd ~/code
# 四路并行，各跑前 3600 帧
./multi_eval configs/test1.json configs/test2.json configs/test3.json configs/test4.json \
    --frames 3600 --out output/full_head4

# 换模型对比（同一批数据）
./multi_eval configs/test1.json configs/test2.json configs/test3.json configs/test4.json \
    --frames 3600 --model /path/to/other.rknn --out output/full_other
```

每个线程创建独立的 `RknnDetector`，按 `i % 3` 绑定到 0/1/2 号 NPU 核。
实测 4 路并行合计 62~73 fps（单路 18~22 fps）。

**`tools/02_diagnose/` 下（源码在 D 盘，要用时传到板子编译）：**

| 工具 | 用途 |
|---|---|
| `ch_scan.cpp` | 打印 boxes 四个通道各自的范围，判断模型有没有坏 |
| `box3.cpp` | 两个模型 + FP32 真值三方对比，逐框算误差 |
| `box_cmp.cpp` | 两个模型互相配对比较（不需要真值） |
| `batch_box_eval.py` | 批量跑多帧，汇总平均误差（调 box3） |
| `dump_truth_boxes.py` | PC 上用 FP32 ONNX 生成真值文件 |
| `trace_layers.py` | 逐层 dump 量化误差，查哪层崩了 |
| `compare_models.py` | **对比两个模型的评测结果**（断号数 + 各线计数），读 multi_eval 的 CSV |
| `show_cfg_scale.py` | 从 quantization.cfg 提每层的 min/max/scale |

板子重启后 `/tmp` 会清空，这些工具要重传重编：

```bash
# 在板子上编译（先把源码传到 /tmp）
g++ -O2 -std=c++17 -I~/code/src /tmp/ch_scan.cpp -o /tmp/ch_scan \
    -lrknnrt $(pkg-config --cflags --libs opencv4)
```

源码在 `D:\multi-road-crossing-vehicle-count\tools\` 对应的 `.cpp` 里
（ch_scan / box_check 需要时我可以重新给）。

