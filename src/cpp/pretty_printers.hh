#pragma once

#include <iostream>
#include <vector>
#include <utility>

// pretty printers
template <typename A, typename B>
static inline std::ostream &
operator<<(std::ostream &o, const std::pair<A, B> &p)
{
  o << "{" << p.first << ":" << p.second << "}";
  return o;
}

template <typename ForwardIterator>
static inline std::ostream &
format(std::ostream &o, ForwardIterator begin, ForwardIterator end)
{
  o << "[";
  bool f = true;
  while (begin != end) {
    if (!f)
      o << ", ";
    f = false;
    o << *begin++;
  }
  o << "]";
  return o;
}

template <typename T, typename Alloc>
static inline std::ostream &
operator<<(std::ostream &o, const std::vector<T, Alloc> &v)
{
  return format(o, v.begin(), v.end());
}
