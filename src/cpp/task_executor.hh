#pragma once

#include <cassert>
#include <memory>
#include <thread>
#include <random>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <atomic>
#include <future>

#if defined(__linux__)
#include <numa.h>
#endif

#include <sched.h>

// task executor object itself is not thread-safe
// non-copyable, non-movable, non-swappable
template <typename T>
class task_executor_thread {
public:

  typedef std::function<T()> func_t;

  // ctor triggers creation of new executor thread
  task_executor_thread(int node = -1) :
    th_(), m_(), cv_(), q_(), running_(true)
  {
    std::thread th(&task_executor_thread<T>::loop, this, node);
    th_ = std::move(th);
  }

  ~task_executor_thread()
  {
    assert(!running_.load());
  }

  task_executor_thread(const task_executor_thread &) = delete;
  task_executor_thread(task_executor_thread &&) = delete;
  task_executor_thread &operator=(const task_executor_thread &) = delete;
  task_executor_thread &operator=(task_executor_thread &&) = delete;

  // blocks until underlying executor thread stops
  void
  shutdown()
  {
    running_.store(false);
    poke();
    th_.join();
  }

  std::future<T>
  enq(func_t fn)
  {
    assert(running_.load());
    std::pair<func_t, std::promise<T>> p(fn, std::move(std::promise<T>()));
    std::future<T> ret = p.second.get_future();
    {
      std::unique_lock<std::mutex> l(m_);
      q_.push(std::move(p));
      cv_.notify_one();
    }
    return std::move(ret);
  }

  std::thread::id
  worker_id() const
  {
    return th_.get_id();
  }

private:

  void
  poke()
  {
    std::unique_lock<std::mutex> l(m_);
    cv_.notify_one();
  }

  void
  loop(int node)
  {
    // thread pinning currently unimplemented for OS-X
#if defined(__linux__)
    if (node != -1) {
      int ret = numa_run_on_node(node);
      if (ret)
        assert(false);
      sched_yield();
    }
#endif

    while (running_.load()) {
      std::unique_lock<std::mutex> l(m_);
      while (q_.empty() && running_.load())
        cv_.wait(l);
      if (q_.empty())
        // done
        break;
      std::pair<func_t, std::promise<T>> work(std::move(q_.front()));
      q_.pop();
      work.second.set_value(work.first());
    }
  }

  std::thread th_;
  std::mutex m_;
  std::condition_variable cv_;
  std::queue<std::pair<func_t, std::promise<T>>> q_;
  std::atomic<bool> running_;
};
