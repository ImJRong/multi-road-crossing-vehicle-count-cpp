# =============================================================================
# 交通流计数 C++ 版 (PC 上调试 / RK3588 上跑, 同一份代码)
#
# 编译:  cmake . && make
# 运行:  ./traffic_count test4
#
# 这个文件里需要改的地方只有三处:
#   1. TRAFFIC_PROJECT_ROOT —— 配置和视频在哪个目录 (换机器才改)
#   2. add_executable 下面的源码列表 —— 新增 .cpp 就加一行
#   3. 想再编一个程序 —— 照抄第 4 节那一段, 改名字和源码
# 其余都不用动.
# =============================================================================

cmake_minimum_required(VERSION 3.16)   # 低于这个版本的 cmake 直接报错, 别猜
project(traffic_count CXX)             # 工程名; CXX 表示只用 C++ 编译器

set(CMAKE_CXX_STANDARD 17)             # 用 C++17 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)    # 编译器不支持 17 就报错, 不允许偷偷降级

# 没有指定编译类型时默认 Release.
# 这一行很关键: Debug 模式不做优化, 推理速度会慢好几倍.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


# ---- 1. 工程根目录 ---------------------------------------------------------
# 配置文件(configs/)和视频(data/videos/)都按这个根目录去找.
# 换电脑、换放的位置, 只改这一行.
# 也可以临时覆盖:  cmake . -DTRAFFIC_PROJECT_ROOT=/别的/路径 && make
set(TRAFFIC_PROJECT_ROOT "/mnt/d/multi-road-crossing-vehicle-count"
    CACHE PATH "工程根目录")


# ---- 2. 找第三方库 ---------------------------------------------------------
# OpenCV: 读视频、缩放、画框画线
find_package(OpenCV REQUIRED)

# ONNX Runtime: 真正跑 .onnx 模型的东西.
# 官方压缩包解压后就是这个目录, 里面有 include/ 和 lib/ 两个子目录.
set(ORT_ROOT "$ENV{HOME}/onnxruntime-linux-x64-1.31.0")


# ---- 3. 编译主程序 ---------------------------------------------------------
# add_executable 后面列出的 .cpp 会被编成一个叫 traffic_count 的可执行文件.
# 新增源文件必须加到这里, 否则写了等于没写.
add_executable(traffic_count
  src/main.cpp        # 入口: 读配置、跑循环、画图
  src/config.cpp      # 解析 configs/*.json
  src/counter.cpp     # 穿线计数
  src/detector.cpp    # ONNX 推理 + 前后处理
)

# 头文件搜索路径. src 放自己的头文件, 这样代码里写 #include "config.hpp" 就能找到
target_include_directories(traffic_count PRIVATE
  src
  "${ORT_ROOT}/include"
  ${OpenCV_INCLUDE_DIRS}
)

# 把根目录的路径编译进程序, 代码里通过 TRAFFIC_PROJECT_ROOT 这个宏拿到.
# 所以 main.cpp 里找配置文件不用传参, 直接拼路径就行.
target_compile_definitions(traffic_count PRIVATE
  TRAFFIC_PROJECT_ROOT="${TRAFFIC_PROJECT_ROOT}")

# 链接库. 漏了 OpenCV 会报 undefined reference to cv::xxx
target_link_libraries(traffic_count PRIVATE
  "${ORT_ROOT}/lib/libonnxruntime.so"
  ${OpenCV_LIBS}
)

# 把库路径写进可执行文件, 运行时不用再 source 环境或设 LD_LIBRARY_PATH.
# 不加这条, 一跑就报 error while loading shared libraries: libonnxruntime.so
set_target_properties(traffic_count PROPERTIES
  BUILD_RPATH "${ORT_ROOT}/lib")


# ---- 4. 以后要加第二个程序, 照抄这一段 -------------------------------------
#
# add_executable(bench
#   src/bench.cpp
#   src/detector.cpp
# )
# target_include_directories(bench PRIVATE src "${ORT_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(bench PRIVATE "${ORT_ROOT}/lib/libonnxruntime.so" ${OpenCV_LIBS})
# set_target_properties(bench PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib")
#
# 编译方式:
#   make           全编 (两个都编)
#   make bench     只编 bench
#   make traffic_count   只编主程序
#
# 注意: 改了 CMakeLists.txt 之后必须先 cmake . 再 make,
# 直接 make 不会重新读这个文件.
