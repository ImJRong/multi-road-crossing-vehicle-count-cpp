// 过线记录写成 CSV：line,group,class,frame,x,y。

#pragma once

#include <cstdio>
#include <string>

#include "counter.hpp"

// path 例: "output/records_test4.csv"
inline void write_records(const std::string& path, const Counter& counter) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "写不了记录文件: %s\n", path.c_str());
    return;
  }
  std::fprintf(f, "line,group,class,frame,x,y\n");
  for (const auto& h : counter.hits()) {
    std::fprintf(f, "%s,%s,%s,%d,%d,%d\n", h.line.c_str(), h.group.c_str(), h.cls.c_str(),
                 h.frame, h.x, h.y);
  }
  std::fclose(f);
  std::printf("记录: %s (%zu 条)\n", path.c_str(), counter.hits().size());
}

