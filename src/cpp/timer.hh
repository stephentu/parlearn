#pragma once

#include <sys/time.h>
#include <cassert>
#include <ctime>
#include <cstdint>
#include <string>
#include <iostream>
#include <macros.hh>

class timer {
private:
  timer(const timer &) = delete;
  timer(timer &&) = delete;
  timer &operator=(const timer &) = delete;

public:

  enum Mode { CLK_GETTIMEOFDAY, CLK_REALTIME };

  timer(Mode m = CLK_GETTIMEOFDAY)
    : m_(m)
  {
    lap();
  }

  inline uint64_t
  elapsed_usec() const
  {
    compiler_barrier();
    const uint64_t t0 = start_;
    const uint64_t t1 = cur_usec(m_);
    compiler_barrier();
    return t1 - t0;
  }

  inline uint64_t
  lap()
  {
    compiler_barrier();
    const uint64_t t0 = start_;
    const uint64_t t1 = cur_usec(m_);
    start_ = t1;
    compiler_barrier();
    return t1 - t0;
  }

  inline uint64_t
  lap_usec()
  {
    return lap();
  }

  inline double
  lap_ms()
  {
    return lap() / 1000.0;
  }

  static inline uint64_t
  cur_usec(Mode m)
  {
    if (m == CLK_GETTIMEOFDAY) {
      struct timeval tv;
      gettimeofday(&tv, 0);
      return ((uint64_t)tv.tv_sec) * 1000000 + tv.tv_usec;
    } else {
      assert(m == CLK_REALTIME);
      struct timespec ts;
      clock_gettime(CLOCK_REALTIME, &ts);
      return ((uint64_t)ts.tv_sec) * 1000000 + (((uint64_t)ts.tv_nsec) / 1000);
    }
  }

private:

  Mode m_;
  uint64_t start_;
};

class scoped_timer {
private:
  timer t;
  std::string region;
  bool enabled;

public:
  scoped_timer(const std::string &region,
               timer::Mode m = timer::CLK_GETTIMEOFDAY,
               bool enabled = true)
    : t(m), region(region), enabled(enabled)
  {}

  ~scoped_timer()
  {
    if (enabled) {
      const double x = t.lap_ms();
      std::cerr << "timed region `" << region << "' took " << x << " ms" << std::endl;
    }
  }
};
