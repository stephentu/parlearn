#include <thread>
#include <vector>
#include <util.hh>
#include <dataset.hh>

using namespace std;
using namespace util;

// read [begin, end) into work, starting from off
template <typename ForwardIterator>
static void
threadwork(std::vector<vec_t> &work,
           size_t off,
           ForwardIterator begin,
           ForwardIterator end)
{
  while (begin != end) {
    work[off++] = *begin;
    ++begin;
  }
}

bool
dataset::do_parallel_materialize()
{
  const auto ncpus = ncpus_online();
  if (x_shape_.first < ncpus)
    // fallback
    return false;
  const size_t bsize = x_shape_.first / ncpus;
  vector<thread> workers;
  vector<vec_t> x(x_shape_.first);
  for (size_t i = 0; i < ncpus; i++) {
    auto begin = x_begin() + (bsize * i);
    auto end =
      ((i+1)==ncpus) ? x_end() : (x_begin() + (bsize * (i+1)));
    workers.emplace_back(threadwork<x_const_iterator>, ref(x), bsize*i, begin, end);
  }
  for (auto &w : workers)
    w.join();
  standard_vec_t y(get_y());
  assert(x.size() == x_shape_.first);
  assert(y.size() == x_shape_.first);
  storage_.reset(new vector_storage(std::move(x), std::move(y)));
  return true;
}
