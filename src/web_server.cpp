#include "web_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
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
  // 忽略掉, 让 send() 老老实实返回 -1, 由我们决定怎么办.
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
  frame_cv_.notify_all();   // 叫醒所有正在等帧的推流线程
  if (accept_thread_.joinable()) accept_thread_.join();
}

// ------------------------------ 帧与状态 ------------------------------

void WebServer::push_frame(const cv::Mat& frame) {
  if (frame.empty()) return;
  std::vector<uchar> buf;
  const std::vector<int> param = {cv::IMWRITE_JPEG_QUALITY, 75};
  if (!cv::imencode(".jpg", frame, buf, param)) return;

  {
    std::lock_guard<std::mutex> lk(frame_mu_);
    jpeg_.swap(buf);
    ++frame_seq_;
  }
  frame_cv_.notify_all();
}

void WebServer::update_stats(const std::vector<LineStat>& lines, int tracks, double fps,
                             double infer_ms, int frame_no) {
  std::ostringstream os;
  os.precision(3);
  os << std::fixed;
  os << "{\"state\":" << state_.load() << ",\"frame\":" << frame_no
     << ",\"tracks\":" << tracks << ",\"fps\":" << fps << ",\"infer\":" << infer_ms
     << ",\"conf\":" << conf_.load() << ",\"iou\":" << iou_.load() << ",\"lines\":[";
  for (size_t i = 0; i < lines.size(); ++i) {
    if (i) os << ',';
    os << "{\"name\":\"" << json_escape(lines[i].name) << "\",\"count\":" << lines[i].count
       << "}";
  }
  os << "]}";
  std::lock_guard<std::mutex> lk(stat_mu_);
  state_json_ = os.str();
}

void WebServer::add_event(const Event& e) {
  std::lock_guard<std::mutex> lk(event_mu_);
  events_.push_back(e);
  while (events_.size() > 200) events_.pop_front();
}

void WebServer::clear_events() {
  std::lock_guard<std::mutex> lk(event_mu_);
  events_.clear();
}

Command WebServer::take_command() {
  const int c = cmd_.exchange(kCmdNone);
  return static_cast<Command>(c);
}

std::string WebServer::start_config() {
  std::lock_guard<std::mutex> lk(cmd_mu_);
  return start_config_;
}

Overlay WebServer::overlay() const {
  const int m = overlay_mask_.load();
  Overlay o;
  o.box = (m & 1) != 0;
  o.id = (m & 2) != 0;
  o.line = (m & 4) != 0;
  o.roi = (m & 8) != 0;
  return o;
}

float WebServer::conf() const { return conf_.load(); }
float WebServer::iou() const { return iou_.load(); }

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
  if (path == "/" || path == "/index.html") {
    serve_index(fd);
  } else if (path == "/stream") {
    serve_stream(fd);
  } else if (path == "/api/state") {
    std::string body;
    {
      std::lock_guard<std::mutex> lk(stat_mu_);
      body = state_json_;
    }
    serve_json(fd, body);
  } else if (path == "/api/configs") {
    std::string body = "{\"configs\":[";
    std::error_code ec;
    std::vector<std::string> names;
    if (fs::is_directory(config_dir_, ec)) {
      for (const auto& e : fs::directory_iterator(config_dir_, ec)) {
        if (!e.is_regular_file()) continue;
        const std::string fn = e.path().filename().string();
        if (fn.size() > 5 && fn.compare(fn.size() - 5, 5, ".json") == 0)
          names.push_back(fn.substr(0, fn.size() - 5));
      }
    }
    std::sort(names.begin(), names.end());
    for (size_t i = 0; i < names.size(); ++i) {
      if (i) body += ',';
      body += "\"" + json_escape(names[i]) + "\"";
    }
    body += "]}";
    serve_json(fd, body);
  } else if (path == "/api/events") {
    std::string body = "{\"events\":[";
    {
      std::lock_guard<std::mutex> lk(event_mu_);
      bool first = true;
      for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
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
    cmd_.store(kCmdStart);
    serve_json(fd, "{\"ok\":1}");
  } else if (path == "/api/stop") {
    cmd_.store(kCmdStop);
    serve_json(fd, "{\"ok\":1}");
  } else if (path == "/api/restart") {
    cmd_.store(kCmdRestart);
    serve_json(fd, "{\"ok\":1}");
  } else if (path == "/api/pause") {
    state_.store(flag_on(query_get(query, "on"), false) ? 2 : 1);
    serve_json(fd, "{\"ok\":1}");
  } else if (path == "/api/param") {
    const std::string c = query_get(query, "conf");
    const std::string i = query_get(query, "iou");
    if (!c.empty()) conf_.store(static_cast<float>(std::atof(c.c_str())));
    if (!i.empty()) iou_.store(static_cast<float>(std::atof(i.c_str())));
    serve_json(fd, "{\"ok\":1}");
  } else if (path == "/api/overlay") {
    Overlay o = overlay();
    o.box = flag_on(query_get(query, "box"), o.box);
    o.id = flag_on(query_get(query, "id"), o.id);
    o.line = flag_on(query_get(query, "line"), o.line);
    o.roi = flag_on(query_get(query, "roi"), o.roi);
    int m = 0;
    if (o.box) m |= 1;
    if (o.id) m |= 2;
    if (o.line) m |= 4;
    if (o.roi) m |= 8;
    overlay_mask_.store(m);
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

void WebServer::serve_stream(int fd) {
  const std::string header =
      "HTTP/1.1 200 OK\r\n"
      "Connection: close\r\n"
      "Cache-Control: no-store\r\n"
      "Pragma: no-cache\r\n"
      "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  if (!send_str(fd, header)) return;

  uint64_t last = 0;
  while (running_) {
    std::vector<uchar> data;
    {
      std::unique_lock<std::mutex> lk(frame_mu_);
      // 最多等 2 秒, 没新帧就再查一次 running_, 免得 stop 之后卡死在这
      frame_cv_.wait_for(lk, std::chrono::seconds(2),
                         [&] { return frame_seq_ != last || !running_; });
      if (!running_) break;
      if (frame_seq_ == last) continue;
      last = frame_seq_;
      data = jpeg_;   // 拷一份再出锁, 别让它占着锁写网络
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
}
