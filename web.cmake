# 网页版 traffic_web 的构建规则.
#
# 不直接写进 CMakeLists.txt, 是为了让主文件只多一行 include(web.cmake),
# 其余照旧 —— 想撤掉的时候删掉那一行就行.
#
# 依赖的变量都在 CMakeLists.txt 里已经定义好了:
#   OpenCV_INCLUDE_DIRS / OpenCV_LIBS   find_package(OpenCV)
#   ORT_ROOT / RKNN_ROOT / USE_RKNN     推理后端
#   TRAFFIC_PROJECT_ROOT                工程根目录
#
# 跑:   ./traffic_web          监听 8080
#       ./traffic_web 9000     换端口
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
  set(WEB_DEFS USE_RKNN)
else()
  set(WEB_DETECTOR_SRC src/detector.cpp)
  set(WEB_BACKEND_INC "${ORT_ROOT}/include")
  set(WEB_BACKEND_LIB "${ORT_ROOT}/lib/libonnxruntime.so")
  set(WEB_BACKEND_RPATH "${ORT_ROOT}/lib")
endif()

add_executable(traffic_web
  src/web_main.cpp        # 入口: 会话管理 + 主循环 + 推帧
  src/web_server.cpp      # HTTP 服务: MJPEG 推流 / 状态 JSON / 控制命令
  src/config.cpp          # 和 traffic_count 共用, 不改
  src/counter.cpp
  src/tracker.cpp
  ${WEB_DETECTOR_SRC}
)

target_include_directories(traffic_web PRIVATE
  src
  ${WEB_BACKEND_INC}
  ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(traffic_web PRIVATE
  ${WEB_DEFS}
  TRAFFIC_PROJECT_ROOT="${TRAFFIC_PROJECT_ROOT}"
  TRAFFIC_WEB_DIR="${CMAKE_CURRENT_SOURCE_DIR}/web/index.html")

target_link_libraries(traffic_web PRIVATE
  ${WEB_BACKEND_LIB}
  ${OpenCV_LIBS}
  Threads::Threads)

set_target_properties(traffic_web PROPERTIES
  BUILD_RPATH "${WEB_BACKEND_RPATH}")

# 全编: make          只编网页版: make traffic_web
