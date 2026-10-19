# =============================================================================
# 交通流计数 C++ 版 (PC 上调试 / RK3588 上跑, 同一份代码)
#
# 编译:  cmake . && make
# 运行:  ./traffic_count test4
#
# 两套推理后端, 用 USE_RKNN 开关切换:
#   默认（OFF）: ONNX Runtime, 在 PC 上跑, 用来调试逻辑
#   -DUSE_RKNN=ON: RKNN, 在板子上跑, 需要 librknnrt.so
#
# 这个文件里需要改的地方只有四处:
#   1. TRAFFIC_PROJECT_ROOT —— 配置和视频在哪个目录 (换机器才改)
#   2. ORT_ROOT / RKNN_ROOT —— 两个推理后端的路径
#   3. add_executable 下面的源码列表 —— 新增 .cpp 就加一行
#   4. 想再编一个程序 —— 照抄第 5 节那一段
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(traffic_count CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 没有指定编译类型时默认 Release.
# Debug 模式不做优化, 推理会慢好几倍.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


# ---- 1. 工程根目录 ---------------------------------------------------------
# 配置文件(configs/)和视频(data/videos/)都按这个根目录找.
set(TRAFFIC_PROJECT_ROOT "/mnt/d/multi-road-crossing-vehicle-count"
    CACHE PATH "工程根目录")


# ---- 2. 推理后端 -----------------------------------------------------------
# 默认用 ONNX Runtime（PC 上调试用）.
option(USE_RKNN "用 RKNN 推理（板子上用）" OFF)

find_package(OpenCV REQUIRED)

# ONNX Runtime: 官方压缩包解压后的目录, 里面有 include/ 和 lib/
set(ORT_ROOT "$ENV{HOME}/onnxruntime-linux-x64-1.31.0")

# RKNN 运行库: 板子的 SDK 里带的, 目录下要有 include/rknn_api.h 和 lib/librknnrt.so
set(RKNN_ROOT "$ENV{HOME}/rknn_runtime" CACHE PATH "librknnrt 所在目录")


# ---- 3. 编译主程序 ---------------------------------------------------------
# 两个后端用不同的源文件, 但 main.cpp / config.cpp / counter.cpp 是共用的.
# 这就是把推理拆成一个类的价值: 换后端只换一个 .cpp.
if(USE_RKNN)
  message(STATUS "推理后端: RKNN (${RKNN_ROOT})")
  add_executable(traffic_count
    src/main.cpp
    src/config.cpp
    src/counter.cpp
    src/rknn_detector.cpp     # RKNN 版: rknn_init / rknn_run / rknn_outputs_get
  )
  target_include_directories(traffic_count PRIVATE
    src
    "${RKNN_ROOT}/include"
    ${OpenCV_INCLUDE_DIRS}
  )
  target_compile_definitions(traffic_count PRIVATE
    USE_RKNN
    TRAFFIC_PROJECT_ROOT="${TRAFFIC_PROJECT_ROOT}")
  target_link_libraries(traffic_count PRIVATE
    "${RKNN_ROOT}/lib/librknnrt.so"
    ${OpenCV_LIBS})
  set_target_properties(traffic_count PROPERTIES
    BUILD_RPATH "${RKNN_ROOT}/lib")
else()
  message(STATUS "推理后端: ONNX Runtime (${ORT_ROOT})")
  add_executable(traffic_count
    src/main.cpp        # 入口: 读配置、跑循环、画图
    src/config.cpp      # 解析 configs/*.json
    src/counter.cpp     # 穿线计数
    src/detector.cpp    # ONNX 推理 + 前后处理
  )
  target_include_directories(traffic_count PRIVATE
    src
    "${ORT_ROOT}/include"
    ${OpenCV_INCLUDE_DIRS}
  )
  target_compile_definitions(traffic_count PRIVATE
    TRAFFIC_PROJECT_ROOT="${TRAFFIC_PROJECT_ROOT}")
  target_link_libraries(traffic_count PRIVATE
    "${ORT_ROOT}/lib/libonnxruntime.so"
    ${OpenCV_LIBS})
  set_target_properties(traffic_count PROPERTIES
    BUILD_RPATH "${ORT_ROOT}/lib")
endif()


# ---- 4. 编译方式 -----------------------------------------------------------
# PC 上（ONNX Runtime）:
#   cmake . && make
#
# 板子上（RKNN）:
#   cmake . -DUSE_RKNN=ON -DRKNN_ROOT=/路径/到/rknn_runtime && make
#
# 换过一次开关之后, 要重新 cmake 一次, 直接 make 不会重读本文件.


# ---- 5. 以后要加第二个程序, 照抄这一段 -------------------------------------
#
# add_executable(bench
#   src/bench.cpp
#   src/detector.cpp
# )
# target_include_directories(bench PRIVATE src "${ORT_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(bench PRIVATE "${ORT_ROOT}/lib/libonnxruntime.so" ${OpenCV_LIBS})
# set_target_properties(bench PROPERTIES BUILD_RPATH "${ORT_ROOT}/lib")
#
# 全编:        make
# 只编一个:    make bench
