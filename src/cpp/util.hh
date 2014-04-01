#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <atomic>
#include <map>
#include <cmath>
#include <type_traits>
#include <macros.hh>
#include <unistd.h>

namespace util {

static inline bool
almost_eq(double a, double b)
{
  return fabs(a - b) <= 1e-5;
}

static inline unsigned
ncpus_online()
{
  return sysconf(_SC_NPROCESSORS_ONLN);
}

// RR nelems amongst nthreads, returns vector of indices
static inline std::vector<std::vector<size_t>>
round_robin(size_t nelems, size_t nthreads)
{
  std::vector<std::vector<size_t>> allocations(nthreads);
  size_t cur = 0;
  for (size_t i = 0; i < nelems; i++) {
    allocations[cur].push_back(i);
    cur = (cur + 1) % nthreads;
  }
  return allocations;
}

static inline std::vector<std::string>
split(const std::string &s, char delim = ' ')
{
  std::vector<std::string> ret;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim))
    ret.push_back(item);
  return ret;
}

template <typename T>
static inline std::string
join(const std::vector<T> &v, const std::string &s)
{
  std::ostringstream oss;
  bool first = true;
  for (auto &t : v) {
    if (!first)
      oss << s;
    first = false;
    oss << t;
  }
  return oss.str();
}

static inline std::vector<size_t>
arange(size_t start, size_t stop, size_t step=1)
{
  std::vector<size_t> ret;
  ret.push_back(start);
  size_t cur = start + step;
  while (cur < stop) {
    ret.push_back(cur);
    cur += step;
  }
  return ret;
}

// returns v[start:end], doesn't deal with negative indices
template <typename T>
static inline std::vector<T>
slice(const std::vector<T> &v, size_t start, size_t end)
{
  return std::vector<T>(v.begin() + start, v.begin() + end);
}

static inline double
sign(double x)
{
  return x >= 0.0 ? 1.0 : -1.0;
}

template <typename T>
static inline std::vector<T>
range(T t)
{
  std::vector<T> ret(t);
  for (T i = 0; i < t; i++)
    ret[i] = i;
  return ret;
}

template <typename T>
static inline std::vector< typename std::enable_if<std::is_floating_point<T>::value, T>::type  >
linspace(T start, T end, size_t n)
{
  ALWAYS_ASSERT(n > 1);
  T spacing = (end - start)/T(n-1);
  std::vector<T> ret(n);
  if (!n)
    return ret;
  ret[0] = start;
  for (size_t i = 1; i < n; i++)
    ret[i] = ret[i-1] + spacing;
  return ret;
}

struct _phelper {
  // lazy man recursion
  template <typename T>
  static std::vector<std::vector<T>>
  product(size_t i, const std::vector<std::vector<T>> &axis)
  {
    if (i == 0) {
      std::vector<std::vector<T>> ret;
      ret.reserve(axis[0].size());
      for (auto &p : axis[0]) {
        std::vector<T> x({p});
        ret.emplace_back(x);
      }
      return ret;
    }
    auto prev = product(i - 1, axis);
    std::vector<std::vector<T>> cur;
    cur.reserve(prev.size() * axis[i].size());
    for (auto &c : axis[i]) {
      for (auto &p : prev) {
        cur.emplace_back(p);
        cur.back().emplace_back(c);
      }
    }
    return cur;
  }
};

template <typename T>
static inline std::vector<std::vector<T>>
product(const std::vector<std::vector<T>> &axis)
{
  if (axis.empty())
    return std::vector<std::vector<T>>();
  return _phelper::product(axis.size() - 1, axis);
}

// XXX: doesn't escape strings...
static inline std::string
smap_to_json(const std::map<std::string, std::string> &m)
{
  std::ostringstream oss;
  oss << "{";
  bool first = true;
  for (auto &kv : m) {
    if (!first)
      oss << ",";
    first = false;
    oss << "\"" << kv.first << "\":\"" << kv.second << "\"";
  }
  oss << "}";
  return oss.str();
}

/**
 * XXX: CoreIDs are not recyclable for now, so NMAXCORES is really the number
 * of threads which can ever be spawned in the system
 */
class core {
public:
  static const unsigned NMaxCores = 512;

  static inline unsigned
  id()
  {
    if (unlikely(tl_core_id == -1)) {
      // initialize per-core data structures
      tl_core_id = g_core_count.fetch_add(1, std::memory_order_acq_rel);
      // did we exceed max cores?
      ALWAYS_ASSERT(unsigned(tl_core_id) < NMaxCores);
    }
    return tl_core_id;
  }

private:
  // the core ID of this core: -1 if not set
  static __thread int tl_core_id;

  // contains a running count of all the cores
  static std::atomic<unsigned> g_core_count;
};

template <typename T>
class padded {
public:
  template <class... Args>
  padded(Args &&... args) : elem_(std::forward<Args>(args)...)
  { }

  // syntactic sugar- can treat like a pointer
  inline T & operator*() { return elem_; }
  inline const T & operator*() const { return elem_; }
  inline T * operator->() { return &elem_; }
  inline const T * operator->() const { return &elem_; }

  inline T & get() { return elem_; }
  inline const T & get() const { return elem_; }

private:
  T elem_;
  CACHE_PADOUT;
};

// requires T to have no-arg ctor
template <typename T>
class percore {
public:

  percore()
  {
    for (size_t i = 0; i < size(); i++)
      new (&(elems()[i])) padded<T>();
  }

  ~percore()
  {
    for (size_t i = 0; i < size(); i++)
      elems()[i].~padded<T>();
  }

  inline T &
  operator[](unsigned i)
  {
    return elems()[i].get();
  }

  inline const T &
  operator[](unsigned i) const
  {
    return elems()[i].get();
  }

  inline T &
  my()
  {
    return (*this)[core::id()];
  }

  inline const T &
  my() const
  {
    return (*this)[core::id()];
  }

  inline size_t
  size() const
  {
    return core::NMaxCores;
  }

protected:

  inline padded<T> *
  elems()
  {
    return (padded<T> *) &bytes_[0];
  }

  inline const padded<T> *
  elems() const
  {
    return (const padded<T> *) &bytes_[0];
  }

  char bytes_[sizeof(padded<T>) * core::NMaxCores];
};

}
