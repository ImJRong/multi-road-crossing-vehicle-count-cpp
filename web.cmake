# 网页版 traffic_web_single 的构建规则.
#
# 不直接写进 CMakeLists.txt, 是为了让主文件只多一行 include(web.cmake),
# 其余配置保持原样; 需要撤销时删除对应行即可.
#
# 依赖的变量都在 CMakeLists.txt 里已经定义好了:
#   OpenCV_INCLUDE_DIRS / OpenCV_LIBS   find_package(OpenCV)
#   ORT_ROOT / RKNN_ROOT / USE_RKNN     推理后端
#   TRAFFIC_PROJECT_ROOT                工程根目录
#
# 跑:   ./traffic_web_single          监听 8080
#       ./traffic_web_single 9000     换端口
# 然后浏览器打开 http://localhost:8080

find_package(Threads REQUIRED)

set(WEB_DETECTOR_SRC "")
set(WEB_BACKEND_INC "")
set(WEB_BACKEND_LIB "")
set(WEB_BACKEND_RPATH "")
set(WEB_DEFS "")

if(USE_RKNN)
  set(WEB_DETECTOR_SRC src/rknn_detector.cpp)
  set(WEB_BACKEND_INC "${RKNN_ROOT}/include")
  set(WEB_BACKEND_LIB "${RKNN_ROOT}/lib/librknnrt.so")
  set(WEB_BACKEND_RPATH "${RKNN_ROOT}/lib")
  # 用 APPEND 而不是 set: 上面已经往 WEB_DEFS 里加了 TRAFFIC_USE_MPP,
  # 这里再 set 会把它冲掉, 编译时那个宏就没了.
  list(APPEND WEB_DEFS USE_RKNN)
else()
  set(WEB_DETECTOR_SRC src/detector.cpp)
  set(WEB_BACKEND_INC "${ORT_ROOT}/include")
  set(WEB_BACKEND_LIB "${ORT_ROOT}/lib/libonnxruntime.so")
  set(WEB_BACKEND_RPATH "${ORT_ROOT}/lib")
endif()

# 视频流硬件解码是可选的: 不开就什么都不加, 编出来的程序和以前一模一样.
# 放在后端判断之后, 用 list(APPEND) 追加, 不会再被上面的 set() 冲掉.
set(WEB_MPP_SRC "")
set(WEB_MPP_LIB "")
if(USE_MPP)
  set(WEB_MPP_SRC src/rkmpp_reader.cpp)
  set(WEB_MPP_LIB "${MPP_ROOT}/build_native/mpp/librockchip_mpp.so")
  list(APPEND WEB_BACKEND_RPATH "${MPP_ROOT}/build_native/mpp")
  list(APPEND WEB_DEFS TRAFFIC_USE_MPP)
  list(APPEND WEB_BACKEND_INC "${MPP_ROOT}/inc" "${MPP_ROOT}/mpp/inc")

  # 解封装用板子自带的 libavformat/libavcodec/libavutil (dev 包已装好).
  # 必须显式链接: OpenCV 的 CMake 配置只带出它自己的依赖, 不会把这几条带出来,
  # 否则链接期报 undefined reference to symbol 'av_packet_free@@LIBAVCODEC_58'.
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(AV REQUIRED libavformat libavcodec libavutil)
  list(APPEND WEB_BACKEND_INC ${AV_INCLUDE_DIRS})
  list(APPEND WEB_MPP_LIB ${AV_LIBRARIES})
  list(APPEND WEB_BACKEND_RPATH ${AV_LIBRARY_DIRS})
  message(STATUS "视频流硬件解码: 开 (${MPP_ROOT}) + libavformat ${AV_libavformat_VERSION}")
else()
  message(STATUS "视频流硬件解码: 关 (只用 OpenCV 软解)")
endif()

add_executable(traffic_web_single
  src/web_main_single.cpp        # 入口: 会话管理 + 主循环 + 推帧
  src/web_server.cpp      # HTTP 服务: MJPEG 推流 / 状态 JSON / 控制命令
  src/config.cpp          # 和 traffic_count 共用, 不改
  src/counter.cpp
  src/tracker.cpp
  ${WEB_DETECTOR_SRC}
  ${WEB_MPP_SRC}
)

target_include_directories(traffic_web_single PRIVATE
  src
  ${WEB_BACKEND_INC}
  ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(traffic_web_single PRIVATE
  ${WEB_DEFS}
  TRAFFIC_PROJECT_ROOT="${TRAFFIC_PROJECT_ROOT}"
  TRAFFIC_WEB_DIR="${CMAKE_CURRENT_SOURCE_DIR}/web/index.html")

target_link_libraries(traffic_web_single PRIVATE
  ${WEB_BACKEND_LIB}
  ${OpenCV_LIBS}
  ${WEB_MPP_LIB}
  Threads::Threads)

set_target_properties(traffic_web_single PROPERTIES
  BUILD_RPATH "${WEB_BACKEND_RPATH}")

# 全编: make          只编网页版: make traffic_web_single

# ---------------------------------------------------------------------------
# 四路网页版 traffic_web_quad
#
# 和 traffic_web_single 共用 web_server.cpp / config / counter / tracker,
# 只是入口换成 web_main_quad.cpp(四路并行 + 循环播 + 按路切换).
# 页面的路径单独给: web/quad.html
# ---------------------------------------------------------------------------
add_executable(traffic_web_quad
  src/web_main_quad.cpp         # 入口: 四路调度(读帧/推理/跟踪三段流水线)
  src/web_server.cpp            # HTTP 服务(多通道版)
  src/config.cpp
  src/counter.cpp
  src/tracker.cpp
  ${WEB_DETECTOR_SRC}
  ${WEB_MPP_SRC}
)

target_include_directories(traffic_web_quad PRIVATE
  src
  ${WEB_BACKEND_INC}
  ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(traffic_web_quad PRIVATE
  ${WEB_DEFS}
  TRAFFIC_PROJECT_ROOT="${TRAFFIC_PROJECT_ROOT}"
  TRAFFIC_WEB_DIR="${CMAKE_CURRENT_SOURCE_DIR}/web/quad.html")

target_link_libraries(traffic_web_quad PRIVATE
  ${WEB_BACKEND_LIB}
  ${OpenCV_LIBS}
  ${WEB_MPP_LIB}
  Threads::Threads)

set_target_properties(traffic_web_quad PROPERTIES
  BUILD_RPATH "${WEB_BACKEND_RPATH}")
