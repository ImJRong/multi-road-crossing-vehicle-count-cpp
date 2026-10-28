#include "web_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

// 把一段字节完整写出去. 客户端中途关页面会写失败, 返回 false 让上层收摊
bool send_all(int fd, const void* data, size_t len) {
  const char* p = static_cast<const char*>(data);
  while (len > 0) {
    const ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL);
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool send_str(int fd, const std::string& s) { return send_all(fd, s.data(), s.size()); }

// 从 a=1&b=2 这种查询串里取一个值
std::string query_get(const std::string& query, const std::string& key) {
  size_t pos = 0;
  while (pos < query.size()) {
    const size_t eq = query.find('=', pos);
    if (eq == std::string::npos) break;
    const size_t amp = query.find('&', eq);
    const std::string k = query.substr(pos, eq - pos);
    const std::string v =
        query.substr(eq + 1, (amp == std::string::npos ? query.size() : amp) - eq - 1);
    if (k == key) return v;
    if (amp == std::string::npos) break;
    pos = amp + 1;
  }
  return std::string();
}

bool flag_on(const std::string& v, bool fallback) {
  if (v.empty()) return fallback;
  return v == "1" || v == "true" || v == "on";
}

// %3A -> ':' , %2F -> '/' , '+' -> ' '
// 浏览器提交 rtsp://... 时会把 : 和 / 都编码掉, 不解码就当不了路径.
// 只给 /api/source 用: 其它接口的参数都是数字或简单名字, 不需要.
std::string url_decode(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '+') {
      out.push_back(' ');
    } else if (s[i] == '%' && i + 2 < s.size()) {
      auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
      };
      const int hi = hex(s[i + 1]);
      const int lo = hex(s[i + 2]);
      if (hi >= 0 && lo >= 0) {
        out.push_back(static_cast<char>(hi * 16 + lo));
        i += 2;
      } else {
        out.push_back(s[i]);
      }
    } else {
      out.push_back(s[i]);
    }
  }
  return out;
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(c);
    } else if (static_cast<unsigned char>(c) < 0x20) {
      out.push_back(' ');
    } else {
      out.push_back(c);
    }
  }
  return out;
}

}  // namespace

WebServer::WebServer(int port, std::string html_path, std::string config_dir,
                     std::string snap_dir)
    : port_(port),
      html_path_(std::move(html_path)),
      config_dir_(std::move(config_dir)),
      snap_dir_(std::move(snap_dir)) {
  std::error_code ec;
  fs::create_directories(snap_dir_, ec);   // 抓拍目录, 已经有就什么都不做
}

WebServer::~WebServer() { stop(); }

void WebServer::start() {
  // 客户端提前关页面时, 往死掉的 socket 写会触发 SIGPIPE, 默认行为是结束进程.
  // 忽略该信号, 使 send() 返回 -1, 由调用方自行处理.
  std::signal(SIGPIPE, SIG_IGN);

  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    std::fprintf(stderr, "socket() 失败\n");
    return;
  }

  int on = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);   // 0.0.0.0, 局域网里的板子也能被访问
  addr.sin_port = htons(static_cast<uint16_t>(port_));

  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::fprintf(stderr, "端口 %d 被占用, 换一个再试\n", port_);
    ::close(listen_fd_);
    listen_fd_ = -1;
    return;
  }
  if (::listen(listen_fd_, 8) < 0) {
    std::fprintf(stderr, "listen() 失败\n");
    ::close(listen_fd_);
    listen_fd_ = -1;
    return;
  }

  running_ = true;
  accept_thread_ = std::thread([this] { accept_loop(); });
  std::printf("网页: http://localhost:%d\n", port_);
}

void WebServer::stop() {
  if (!running_.exchange(false)) return;
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  for (int i = 0; i < kMaxCams; ++i) frame_cv_[i].notify_all();   // 叫醒所有正在等帧的推流线程
  if (accept_thread_.joinable()) accept_thread_.join();
}

// ------------------------------ 帧与状态 ------------------------------

void WebServer::push_frame(const cv::Mat& frame, int cam) {
  if (frame.empty()) return;
  if (cam < 0 || cam >= kMaxCams) return;

  // 节流: 没人看这一路就什么都不做, 连 JPEG 编码都省掉
  const bool need_full = full_viewers_[cam].load() > 0;
  const bool need_thumb = thumb_viewers_[cam].load() > 0;
  if (!need_full && !need_thumb) return;

  {
    std::lock_guard<std::mutex> lk(frame_mu_[cam]);
    if (need_full) {
      std::vector<uchar> buf;
      const std::vector<int> param = {cv::IMWRITE_JPEG_QUALITY, 75};
      if (cv::imencode(".jpg", frame, buf, param)) {
        jpeg_[cam].swap(buf);
      }
    }
    if (need_thumb) {
      // 缩略图: 宽度压到 kThumbWidth, 流量降 5 倍多
      cv::Mat small;
      const double sc = static_cast<double>(kThumbWidth) / frame.cols;
      if (sc < 1.0) {
        cv::resize(frame, small, cv::Size(kThumbWidth,
                    std::max(1, static_cast<int>(frame.rows * sc))), cv::INTER_AREA);
      } else {
        small = frame;
      }
      std::vector<uchar> buf;
      const std::vector<int> param = {cv::IMWRITE_JPEG_QUALITY, 68};
      if (cv::imencode(".jpg", small, buf, param)) {
        thumb_jpeg_[cam].swap(buf);
      }
    }
    ++frame_seq_[cam];
  }
  frame_cv_[cam].notify_all();
}

void WebServer::sample_sys() {
  // CPU：读 /proc/stat 第一行，用两次采样的差值算占用率
  std::ifstream f("/proc/stat");
  if (f) {
    std::string tag;
    unsigned long long user = 0, nice = 0, sys = 0, idle = 0;
    unsigned long long iowait = 0, irq = 0, softirq = 0, steal = 0;
    f >> tag >> user >> nice >> sys >> idle >> iowait >> irq >> softirq >> steal;
    const unsigned long long total = user + nice + sys + idle + iowait + irq + softirq + steal;
    const unsigned long long idle_all = idle + iowait;
    if (last_cpu_total_ && total > last_cpu_total_) {
      const double dt = static_cast<double>(total - last_cpu_total_);
      const double di = static_cast<double>(idle_all - last_cpu_idle_);
      cpu_usage_ = 100.0 * (1.0 - di / dt);
      if (cpu_usage_ < 0.0) cpu_usage_ = 0.0;
      if (cpu_usage_ > 100.0) cpu_usage_ = 100.0;
    }
    last_cpu_total_ = total;
    last_cpu_idle_ = idle_all;
  }

  // 温度（毫摄氏度 -> 摄氏度）
  {
    std::ifstream tf("/sys/class/thermal/thermal_zone0/temp");
    long milli = 0;
    if (tf >> milli) temp_c_ = milli / 1000.0;
  }

  // NPU / GPU 当前频率（Hz -> MHz）
  auto read_mhz = [](const char* path) -> int {
    std::ifstream f(path);
    long hz = 0;
    if (f >> hz) return static_cast<int>(hz / 1000000);
    return 0;
  };
  npu_mhz_ = read_mhz("/sys/class/devfreq/fdab0000.npu/cur_freq");
  gpu_mhz_ = read_mhz("/sys/class/devfreq/fb000000.gpu/cur_freq");

  // NPU 三核负载（格式: NPU load:  Core0: 46%, Core1:  8%, Core2:  8%,）
  // 需要 /sys/kernel/debug 对普通用户可读，否则读失败就保持 0
  {
    std::ifstream nf("/sys/kernel/debug/rknpu/load");
    std::string line;
    if (std::getline(nf, line)) {
      auto pick = [&line](const char* key) -> int {
        const char* p = std::strstr(line.c_str(), key);
        if (!p) return 0;
        return std::atoi(p + std::strlen(key));
      };
      npu_c0_ = pick("Core0:");
      npu_c1_ = pick("Core1:");
      npu_c2_ = pick("Core2:");
    }
  }
}

void WebServer::update_stats(const std::vector<LineStat>& lines, int tracks, double fps,
                             double infer_ms, int frame_no, int cam) {
  std::ostringstream os;
  os.precision(3);
  os << std::fixed;
  os << "{\"state\":" << state_.load() << ",\"frame\":" << frame_no
     << ",\"tracks\":" << tracks << ",\"fps\":" << fps << ",\"infer\":" << infer_ms
     << ",\"conf\":" << conf_[cam].load() << ",\"iou\":" << iou_[cam].load() << ",\"lines\":[";
  for (size_t i = 0; i < lines.size(); ++i) {
    if (i) os << ',';
    os << "{\"name\":\"" << json_escape(lines[i].name) << "\",\"count\":" << lines[i].count
       << "}";
  }
  os << "]}";
  std::lock_guard<std::mutex> lk(stat_mu_);
  state_json_[cam] = os.str();
}

void WebServer::add_event(const Event& e, int cam) {
  if (cam < 0 || cam >= kMaxCams) return;
  std::lock_guard<std::mutex> lk(event_mu_);
  events_[cam].push_back(e);
  while (events_[cam].size() > 200) events_[cam].pop_front();
}

void WebServer::clear_events(int cam) {
  if (cam < 0 || cam >= kMaxCams) return;
  std::lock_guard<std::mutex> lk(event_mu_);
  events_[cam].clear();
}

Command WebServer::take_command(int cam) {
  if (cam < 0 || cam >= kMaxCams) return kCmdNone;
  return static_cast<Command>(cmd_[cam].exchange(kCmdNone));
}

std::string WebServer::start_config() {
  std::lock_guard<std::mutex> lk(cmd_mu_);
  return start_config_;
}

Overlay WebServer::overlay(int cam) const {
  if (cam < 0 || cam >= kMaxCams) return Overlay();
  const int m = overlay_mask_[cam].load();
  Overlay o;
  o.box = (m & 1) != 0;
  o.id = (m & 2) != 0;
  o.line = (m & 4) != 0;
  o.roi = (m & 8) != 0;
  return o;
}

float WebServer::conf(int cam) const {
  if (cam < 0 || cam >= kMaxCams) return 0.3f;
  return conf_[cam].load();
}
float WebServer::iou(int cam) const {
  if (cam < 0 || cam >= kMaxCams) return 0.5f;
  return iou_[cam].load();
}

// 只在启动时由 main 调一次, 之后网页改值走 /api/param, 不再走这里.
// 单独封装为函数而非复用 /api/param 的解析逻辑, 原因:
// 一是从 float 转字符串再转回来没必要, 二是不想让 HTTP 解析路径沾上初始化.
void WebServer::init_params(int cam, float conf, float iou) {
  if (cam < 0 || cam >= kMaxCams) return;
  if (conf > 0.0f) conf_[cam].store(conf);   // 配置里漏写(<=0)就保留默认值
  if (iou > 0.0f) iou_[cam].store(iou);
}

// ---------------- 视频源 ----------------

void WebServer::set_source(int cam, const std::string& url, const std::string& cfg) {
  if (cam < 0 || cam >= kMaxCams) return;
  std::lock_guard<std::mutex> lk(src_mu_);
  source_[cam] = url;
  if (!cfg.empty()) src_cfg_[cam] = cfg;
}

std::string WebServer::source(int cam) const {
  if (cam < 0 || cam >= kMaxCams) return std::string();
  std::lock_guard<std::mutex> lk(src_mu_);
  return source_[cam];
}

std::string WebServer::source_config(int cam) const {
  if (cam < 0 || cam >= kMaxCams) return std::string();
  std::lock_guard<std::mutex> lk(src_mu_);
  return src_cfg_[cam];
}

int WebServer::source_gen(int cam) const {
  if (cam < 0 || cam >= kMaxCams) return 0;
  return src_gen_[cam].load();
}

void WebServer::bump_source_gen(int cam) {
  if (cam < 0 || cam >= kMaxCams) return;
  src_gen_[cam].fetch_add(1);
}

void WebServer::set_idle(int cam, bool v) {
  if (cam < 0 || cam >= kMaxCams) return;
  idle_[cam].store(v);
}

// 断开只停这一路, 不清 source_: 网页刷新后还是原来填的那一路, 再点接入就回来.
void WebServer::request_stop(int cam) {
  if (cam < 0 || cam >= kMaxCams) return;
  src_stop_[cam].store(true);
}

bool WebServer::take_stop(int cam) {
  if (cam < 0 || cam >= kMaxCams) return false;
  return src_stop_[cam].exchange(false);
}

// data/videos 下的视频文件, 返回能直接填进输入框的相对路径.
// 只列 mp4/avi/mkv/mov, 按名字排. 目录不存在就返回空, 界面上就是空列表.
std::vector<std::string> WebServer::video_list() const {
  std::vector<std::string> out;
  std::error_code ec;
  const fs::path dir = fs::path(config_dir_).parent_path() / "data" / "videos";
  if (!fs::is_directory(dir, ec)) return out;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    if (!e.is_regular_file()) continue;
    const std::string fn = e.path().filename().string();
    const size_t dot = fn.find_last_of('.');
    if (dot == std::string::npos) continue;
    std::string ext = fn.substr(dot);
    for (auto& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (ext != ".mp4" && ext != ".avi" && ext != ".mkv" && ext != ".mov") continue;
    out.push_back("data/videos/" + fn);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::string> WebServer::config_list() const {
  std::vector<std::string> names;
  std::error_code ec;
  if (!fs::is_directory(config_dir_, ec)) return names;
  for (const auto& e : fs::directory_iterator(config_dir_, ec)) {
    if (!e.is_regular_file()) continue;
    const std::string fn = e.path().filename().string();
    if (fn.size() > 5 && fn.compare(fn.size() - 5, 5, ".json") == 0)
      names.push_back(fn.substr(0, fn.size() - 5));
  }
  std::sort(names.begin(), names.end());
  return names;
}

// ------------------------------ 监听与分发 ------------------------------

void WebServer::accept_loop() {
  while (running_) {
    const int fd = ::accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) {
      if (!running_) break;
      continue;
    }
    std::thread(&WebServer::handle_client, this, fd).detach();
  }
}

void WebServer::handle_client(int fd) {
  // 读请求头. 浏览器一次就发完, 但要防着只发一半的情况
  std::string head;
  char buf[2048];
  while (head.find("\r\n\r\n") == std::string::npos && head.size() < 8192) {
    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n <= 0) {
      ::close(fd);
      return;
    }
    head.append(buf, static_cast<size_t>(n));
  }

  // 第一行形如: GET /api/state?x=1 HTTP/1.1
  std::string target;
  if (head.compare(0, 4, "GET ") == 0) {
    const size_t sp = head.find(' ', 4);
    if (sp != std::string::npos) target = head.substr(4, sp - 4);
  }
  if (target.empty()) {
    const std::string r = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
    send_str(fd, r);
    ::close(fd);
    return;
  }

  std::string path = target;
  std::string query;
  const size_t q = target.find('?');
  if (q != std::string::npos) {
    path = target.substr(0, q);
    query = target.substr(q + 1);
  }

  route(fd, "GET", path, query);

  // /stream 是长连接, 里面自己管关闭; 这里再 close 一次也无害
  ::close(fd);
}

void WebServer::route(int fd, const std::string& /*method*/, const std::string& path,
                      const std::string& query) {
  // cam = 第几路, 缺省 0(单路版行为不变)
  const int cam_arg = std::atoi(query_get(query, "cam").c_str());
  const int cam = (cam_arg < 0 || cam_arg >= kMaxCams) ? 0 : cam_arg;

  if (path == "/" || path == "/index.html") {
    serve_index(fd);
  } else if (path == "/stream") {
    const std::string th = query_get(query, "thumb");
    const bool thumb = !th.empty() && th != "0";
    serve_stream(fd, cam, thumb);
  } else if (path == "/api/state") {
    std::string body;
    {
      // 每次请求都重新采样，这样暂停时 CPU/温度/频率仍在刷新
      sample_sys();
      std::string j;
      {
        std::lock_guard<std::mutex> lk(stat_mu_);
        j = state_json_[cam];
      }
      // state 用实时值覆盖: 结束后没人再 update_stats, JSON 里是旧的
      {
        char stbuf[16];
        std::snprintf(stbuf, sizeof(stbuf), "%d", state_.load());
        const size_t sp = j.find("\"state\":");
        if (sp != std::string::npos) {
          const size_t ep = j.find(',', sp);
          if (ep != std::string::npos) j.replace(sp + 8, ep - sp - 8, stbuf);
        }
      }
      std::ostringstream sys;
      sys.precision(2);
      sys << std::fixed;
      std::string src_url, src_cfg;
      {
        std::lock_guard<std::mutex> lk(src_mu_);
        src_url = source_[cam];
        src_cfg = src_cfg_[cam];
      }
      sys << "\"cpu\":" << cpu_usage_ << ",\"temp\":" << temp_c_
          << ",\"npu\":" << npu_mhz_ << ",\"gpu\":" << gpu_mhz_
          << ",\"npu_load\":[" << npu_c0_ << "," << npu_c1_ << "," << npu_c2_ << "]"
          << ",\"idle\":" << (idle_[cam].load() ? 1 : 0)
          << ",\"source\":\"" << json_escape(src_url) << "\""
          << ",\"config\":\"" << json_escape(src_cfg) << "\"";
      if (j.size() > 2 && j.back() == '}') {
        // 在最后一个 } 前插入
        j.insert(j.size() - 1, "," + sys.str());
      } else {
        j = "{" + sys.str() + "}";
      }
      body = j;
    }
    serve_json(fd, body);
  } else if (path == "/api/configs") {
    std::string body = "{\"configs\":[";
    const std::vector<std::string> names = config_list();
    for (size_t i = 0; i < names.size(); ++i) {
      if (i) body += ',';
      body += "\"" + json_escape(names[i]) + "\"";
    }
    body += "]}";
    serve_json(fd, body);
  } else if (path == "/api/videos") {
    std::string body = "{\"videos\":[";
    const std::vector<std::string> vids = video_list();
    for (size_t i = 0; i < vids.size(); ++i) {
      if (i) body += ',';
      body += "\"" + json_escape(vids[i]) + "\"";
    }
    body += "]}";
    serve_json(fd, body);
  } else if (path == "/api/source") {
    // 换源/换配置: 只存字符串 + 版本号 +1, 真正的起停交给主线程.
    // stop=1 只停这一路, 地址和配置原样留着; 否则按 url 换源(空 = 也停, 兼容老用法).
    if (flag_on(query_get(query, "stop"), false)) {
      request_stop(cam);
    } else {
      set_source(cam, url_decode(query_get(query, "url")),
                 url_decode(query_get(query, "config")));
      bump_source_gen(cam);
    }
    serve_json(fd, "{\"ok\":1}");
  } else if (path == "/api/events") {
    std::string body = "{\"events\":[";
    {
      std::lock_guard<std::mutex> lk(event_mu_);
      bool first = true;
      for (auto it = events_[cam].rbegin(); it != events_[cam].rend(); ++it) {
        if (!first) body += ',';
        first = false;
        char ts[32];
        std::snprintf(ts, sizeof(ts), "%.1f", it->ts);
        body += "{\"frame\":" + std::to_string(it->frame) + ",\"line\":\"" +
                json_escape(it->line) + "\",\"cls\":\"" + json_escape(it->cls) +
                "\",\"x\":" + std::to_string(it->x) + ",\"y\":" + std::to_string(it->y) +
                ",\"ts\":" + ts + ",\"snap\":\"" + json_escape(it->snap) + "\"}";
      }
    }
    body += "]}";
    serve_json(fd, body);
  } else if (path == "/api/start") {
    const std::string cfg = query_get(query, "config");
    {
      std::lock_guard<std::mutex> lk(cmd_mu_);
      start_config_ = cfg;
    }
    cmd_[cam].store(kCmdStart);
    serve_json(fd, "{\"ok\":1}");
  } else if (path == "/api/stop") {
    cmd_[cam].store(kCmdStop);
    serve_json(fd, "{\"ok\":1}");
  } else if (path == "/api/restart") {
    cmd_[cam].store(kCmdRestart);
    serve_json(fd, "{\"ok\":1}");
  } else if (path == "/api/pause") {
    state_.store(flag_on(query_get(query, "on"), false) ? 2 : 1);
    serve_json(fd, "{\"ok\":1}");
  } else if (path == "/api/param") {
    const std::string c = query_get(query, "conf");
    const std::string i = query_get(query, "iou");
    if (!c.empty()) conf_[cam].store(static_cast<float>(std::atof(c.c_str())));
    if (!i.empty()) iou_[cam].store(static_cast<float>(std::atof(i.c_str())));
    serve_json(fd, "{\"ok\":1}");
  } else if (path == "/api/quit") {
    cmd_[0].store(kCmdQuit);   // 全局命令, 只走 0 号位
    serve_json(fd, "{\"ok\":1}");
  } else if (path == "/api/overlay") {
    Overlay o = overlay(cam);
    o.box = flag_on(query_get(query, "box"), o.box);
    o.id = flag_on(query_get(query, "id"), o.id);
    o.line = flag_on(query_get(query, "line"), o.line);
    o.roi = flag_on(query_get(query, "roi"), o.roi);
    int m = 0;
    if (o.box) m |= 1;
    if (o.id) m |= 2;
    if (o.line) m |= 4;
    if (o.roi) m |= 8;
    overlay_mask_[cam].store(m);
    serve_json(fd, "{\"ok\":1}");
  } else if (path.compare(0, 6, "/snap/") == 0) {
    serve_snap(fd, path.substr(6));
  } else {
    const std::string r =
        "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain; charset=utf-8\r\n"
        "Content-Length: 9\r\n\r\nnot found";
    send_str(fd, r);
  }
}

void WebServer::serve_index(int fd) {
  std::ifstream in(html_path_, std::ios::binary);
  if (!in) {
    const std::string msg = "找不到页面文件: " + html_path_ +
                            "\n（web/index.html 要和可执行文件在同一台机器上）";
    std::string r = "HTTP/1.1 500 Internal Server Error\r\n"
                    "Content-Type: text/plain; charset=utf-8\r\nContent-Length: " +
                    std::to_string(msg.size()) + "\r\n\r\n" + msg;
    send_str(fd, r);
    return;
  }
  std::string html((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::string r = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                  "Cache-Control: no-store\r\nContent-Length: " +
                  std::to_string(html.size()) + "\r\n\r\n" + html;
  send_str(fd, r);
}

void WebServer::serve_json(int fd, const std::string& body) {
  std::string r = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\n"
                  "Cache-Control: no-store\r\nContent-Length: " +
                  std::to_string(body.size()) + "\r\n\r\n" + body;
  send_str(fd, r);
}

void WebServer::serve_snap(int fd, const std::string& name) {
  // 只允许纯文件名, 挡住 ../ 这种路径穿越
  if (name.find('/') != std::string::npos || name.find("..") != std::string::npos) {
    serve_json(fd, "{\"error\":\"bad name\"}");
    return;
  }
  const std::string full = snap_dir_ + "/" + name;
  std::ifstream in(full, std::ios::binary);
  if (!in) {
    const std::string r = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
    send_str(fd, r);
    return;
  }
  std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::string r = "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                  std::to_string(data.size()) + "\r\n\r\n" + data;
  send_str(fd, r);
}

void WebServer::serve_stream(int fd, int cam, bool thumb) {
  const std::string header =
      "HTTP/1.1 200 OK\r\n"
      "Connection: close\r\n"
      "Cache-Control: no-store\r\n"
      "Pragma: no-cache\r\n"
      "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  // 先记下"有人在看", 再发 header.
  // 反过来的话, 头几帧会因为没人看而不编码, 客户端要等下一帧才有画面.
  if (thumb) ++thumb_viewers_[cam];
  else ++full_viewers_[cam];

  if (!send_str(fd, header)) {
    if (thumb) --thumb_viewers_[cam];
    else --full_viewers_[cam];
    return;
  }

  uint64_t last = 0;
  while (running_) {
    std::vector<uchar> data;
    {
      std::unique_lock<std::mutex> lk(frame_mu_[cam]);
      // 最多等 2 秒, 没新帧就再查一次 running_, 免得 stop 之后卡死在这
      frame_cv_[cam].wait_for(lk, std::chrono::seconds(2),
                              [&] { return frame_seq_[cam] != last || !running_; });
      if (!running_) break;
      if (frame_seq_[cam] == last) continue;
      last = frame_seq_[cam];
      data = thumb ? thumb_jpeg_[cam] : jpeg_[cam];   // 拷一份再出锁
    }
    if (data.empty()) continue;

    char part[128];
    const int n = std::snprintf(part, sizeof(part),
                                "--frame\r\nContent-Type: image/jpeg\r\n"
                                "Content-Length: %zu\r\n\r\n",
                                data.size());
    if (!send_all(fd, part, static_cast<size_t>(n))) break;
    if (!send_all(fd, data.data(), data.size())) break;
    if (!send_str(fd, "\r\n")) break;
  }
  if (thumb) --thumb_viewers_[cam];
  else --full_viewers_[cam];
}
