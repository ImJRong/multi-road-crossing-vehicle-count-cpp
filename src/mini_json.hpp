// 够用的小 JSON 解析器, 只干配置文件这一件事.
// 不拉第三方库: 板子上要能编译, 少一个依赖少一份麻烦.
#pragma once

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini {

struct Value {
  enum Type { NUL, BOOL, NUM, STR, ARR, OBJ };

  Type type = NUL;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Value> arr;
  std::map<std::string, Value> obj;

  bool is_obj() const { return type == OBJ; }
  bool is_arr() const { return type == ARR; }

  bool has(const std::string& key) const {
    return type == OBJ && obj.find(key) != obj.end();
  }

  double get_num(const std::string& key, double def) const {
    if (!has(key) || obj.at(key).type != NUM) return def;
    return obj.at(key).num;
  }

  int get_int(const std::string& key, int def) const {
    return static_cast<int>(get_num(key, def));
  }

  bool get_bool(const std::string& key, bool def) const {
    if (!has(key) || obj.at(key).type != BOOL) return def;
    return obj.at(key).b;
  }

  std::string get_str(const std::string& key, const std::string& def) const {
    if (!has(key) || obj.at(key).type != STR) return def;
    return obj.at(key).str;
  }
};

class Parser {
 public:
  explicit Parser(const std::string& text) : s_(text) {}

  Value parse() {
    skip();
    Value v = value();
    skip();
    return v;
  }

 private:
  const std::string& s_;
  size_t i_ = 0;

  [[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("JSON 解析失败(位置 " + std::to_string(i_) + "): " + msg);
  }

  void skip() {
    while (i_ < s_.size() &&
           (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) {
      ++i_;
    }
  }

  char peek() { return i_ < s_.size() ? s_[i_] : '\0'; }

  void expect(char c) {
    if (peek() != c) fail(std::string("期待 '") + c + "'");
    ++i_;
  }

  Value value() {
    skip();
    char c = peek();
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') {
      Value v;
      v.type = Value::STR;
      v.str = string();
      return v;
    }
    Value v;
    if (c == 't') {
      literal("true");
      v.type = Value::BOOL;
      v.b = true;
      return v;
    }
    if (c == 'f') {
      literal("false");
      v.type = Value::BOOL;
      v.b = false;
      return v;
    }
    if (c == 'n') {
      literal("null");
      return v;
    }
    return number();
  }

  bool literal(const std::string& word) {
    if (s_.compare(i_, word.size(), word) != 0) return false;
    i_ += word.size();
    return true;
  }

  Value object() {
    Value v;
    v.type = Value::OBJ;
    expect('{');
    skip();
    if (peek() == '}') {
      ++i_;
      return v;
    }
    while (true) {
      skip();
      std::string key = string();
      skip();
      expect(':');
      v.obj[key] = value();
      skip();
      if (peek() == ',') {
        ++i_;
        continue;
      }
      expect('}');
      return v;
    }
  }

  Value array() {
    Value v;
    v.type = Value::ARR;
    expect('[');
    skip();
    if (peek() == ']') {
      ++i_;
      return v;
    }
    while (true) {
      v.arr.push_back(value());
      skip();
      if (peek() == ',') {
        ++i_;
        continue;
      }
      expect(']');
      return v;
    }
  }

  std::string string() {
    expect('"');
    std::string out;
    while (i_ < s_.size()) {
      char c = s_[i_++];
      if (c == '"') return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i_ >= s_.size()) break;
      char e = s_[i_++];
      switch (e) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'u':
          if (i_ + 4 <= s_.size()) i_ += 4;
          out += '?';
          break;
        default: out += e; break;
      }
    }
    fail("字符串没闭合");
  }

  Value number() {
    size_t start = i_;
    if (peek() == '-' || peek() == '+') ++i_;
    while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) ||
                              s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E' ||
                              s_[i_] == '-' || s_[i_] == '+')) {
      ++i_;
    }
    if (start == i_) fail("这里应该是个数字");
    Value v;
    v.type = Value::NUM;
    v.num = std::atof(s_.substr(start, i_ - start).c_str());
    return v;
  }
};

inline Value parse(const std::string& text) { return Parser(text).parse(); }

inline Value parse_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("打不开配置文件: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return parse(ss.str());
}

}  // namespace mini
