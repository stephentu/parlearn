#pragma once

#include <macros.hh>
#include <vec.hh>
#include <loss_functions.hh>
#include <dataset.hh>

#include <thread>
#include <limits>
#include <tbb/concurrent_queue.h>

namespace model {

template <typename ForwardIterator>
static inline standard_vec_t
linear_Ax(ForwardIterator begin, ForwardIterator end, const standard_vec_t &x)
{
  standard_vec_t b;
  b.reserve(end - begin);
  while (begin != end) {
    b.push_back(ops::dot(*begin, x));
    ++begin;
  }
  return b;
}

template <typename LossFunc>
class linear_model {
public:

  typedef LossFunc loss_function_type;
  static constexpr const double norm_bound_const = 1.0;

  linear_model(double lambda,
               LossFunc lossfn = LossFunc())
    : lambda_(lambda),
      w_(),
      lossfn_(lossfn),
      nthreads_(4)
  {}

  linear_model(double lambda,
               const standard_vec_t &w,
               LossFunc lossfn = LossFunc())
    : lambda_(lambda),
      w_(w),
      lossfn_(lossfn),
      nthreads_(4)
  { }

  linear_model(double lambda,
               standard_vec_t &&w,
               LossFunc lossfn = LossFunc())
    : lambda_(lambda),
      w_(std::move(w)),
      lossfn_(lossfn),
      nthreads_(4)
  { }

  ~linear_model()
  {
    shutdown_pool();
  }

  linear_model(const linear_model &that)
  {
    shutdown_pool();
    lambda_ = that.lambda_;
    w_ = that.w_;
    lossfn_ = that.lossfn_;
    nthreads_ = that.nthreads_;
  }

  // linear model API

  inline void
  set_nthreads(size_t nthreads)
  {
    ALWAYS_ASSERT(thds_.empty());
    ALWAYS_ASSERT(nthreads > 0);
    nthreads_ = nthreads;
  }

  inline size_t get_nthreads() const { return nthreads_; }

private:
  struct message {
    const standard_vec_t *w_;
    const dataset *d_;
    size_t start_;
    size_t end_;
  };

public:

  inline double
  parallel_empirical_risk(const standard_vec_t &w, const dataset &d) const
  {
    const size_t n = d.get_x_shape().first;
    const size_t elems_per_thread = n / nthreads_;
    if (unlikely(elems_per_thread == 0))
      return empirical_risk(w, d);
    ensure_pool();
    for (size_t i = 0; i < nthreads_; i++) {
      messages_[i].w_ = &w;
      messages_[i].d_ = &d;
      messages_[i].start_ = i*elems_per_thread;
      messages_[i].end_ = (i+1)==nthreads_ ? n : (i+1)*elems_per_thread;
      inqueues_[i].push(&messages_[i]);
    }
    double accum = 0.0;
    for (auto &q : outqueues_) {
      double s;
      q.pop(s);
      accum += s;
    }
    accum /= double(n);
    accum += lambda_ / 2.0 * ops::dot(w, w);
    return accum;
  }

  /**
   * Evaluates the objective function on d, F(D)
   */
  inline double
  empirical_risk(const standard_vec_t &w, const dataset &d, size_t start, size_t end) const
  {
    const size_t n = end - start;
    double sum_loss = 0.0;
    const auto it_end = d.begin() + end;
    for (auto it = d.begin() + start; it != it_end; ++it)
      sum_loss += lossfn_.loss(*it.second(), ops::dot(w, *it.first()));
    return 1.0 / double(n) * sum_loss + lambda_ / 2.0 * ops::dot(w, w);
  }

  inline double
  empirical_risk(const standard_vec_t &w, const dataset &d) const
  {
    return empirical_risk(w, d, 0, d.get_x_shape().first);
  }

  inline double
  empirical_risk(const dataset &d) const
  {
    return empirical_risk(w_, d);
  }

  inline void
  inplace_grad_empirical_risk(
      standard_vec_t &grad,
      const standard_vec_t &w,
      const dataset &d,
      size_t start,
      size_t end) const
  {
    const size_t n = end - start;
    grad.resize(w.size());
    grad.zero();
    const auto it_end = d.begin() + end;
    for (auto it = d.begin() + start; it != it_end; ++it) {
      const auto &x = *it.first();
      const double dloss = lossfn_.dloss(*it.second(), ops::dot(w, x));
      const auto inner_it_end = x.end();
      for (auto inner_it = x.begin();
           inner_it != inner_it_end; ++inner_it) {
        const size_t feature_idx = inner_it.tell();
        grad[feature_idx] += (*inner_it) * dloss;
      }
    }
    grad *= (1.0 / double(n));
    grad.add(lambda_, w);
  }

  inline standard_vec_t
  grad_empirical_risk(const standard_vec_t &w, const dataset &d, size_t start, size_t end) const
  {
    const size_t n = end - start;
    standard_vec_t term1(w.size());
    const auto it_end = d.begin() + end;
    for (auto it = d.begin() + start; it != it_end; ++it) {
      const auto &x = *it.first();
      const double dloss = lossfn_.dloss(*it.second(), ops::dot(w, x));
      const auto inner_it_end = x.end();
      for (auto inner_it = x.begin();
           inner_it != inner_it_end; ++inner_it) {
        const size_t feature_idx = inner_it.tell();
        term1[feature_idx] += (*inner_it) * dloss;
      }
    }
    return 1.0 / double(n) * term1 + lambda_ * w;
  }

  inline standard_vec_t
  grad_empirical_risk(const standard_vec_t &w, const dataset &d) const
  {
    return grad_empirical_risk(w, d, 0, d.get_x_shape().first);
  }

  inline standard_vec_t
  grad_empirical_risk(const dataset &d) const
  {
    return grad_empirical_risk(w_, d);
  }

  inline double
  norm_grad_empirical_risk(const standard_vec_t &w, const dataset &d) const
  {
    return grad_empirical_risk(w, d).norm();
  }

  inline double
  norm_grad_empirical_risk(const dataset &d) const
  {
    return norm_grad_empirical_risk(w_, d);
  }

  inline dataset
  transform(const dataset &d) const
  {
    return d;
  }

  inline standard_vec_t
  predict(const dataset &d) const
  {
    return linear_Ax(d.x_begin(), d.x_end(), w_).sign();
  }

  inline double get_lambda() const { return lambda_; }
  inline standard_vec_t &weightvec() { return w_; }
  inline const standard_vec_t & weightvec() const { return w_; }
  inline const LossFunc & get_lossfn() const { return lossfn_; }

  inline linear_model<LossFunc>
  buildfrom(const standard_vec_t &w) const
  {
    return linear_model<LossFunc>(lambda_, w, lossfn_);
  }

  inline linear_model<LossFunc>
  buildfrom(standard_vec_t &&w) const
  {
    return linear_model<LossFunc>(lambda_, std::move(w), lossfn_);
  }

  inline std::map<std::string, std::string>
  mapconfig() const
  {
    std::map<std::string, std::string> m;
    m["model_type"]   = "linear";
    m["model_lambda"] = std::to_string(lambda_);
    return m;
  }

protected:

  inline double
  task(const standard_vec_t &w,
       const dataset &d,
       size_t start,
       size_t end) const
  {
    double sum_loss = 0.0;
    const auto it_end = d.begin() + end;
    for (auto it = d.begin() + start; it != it_end; ++it)
      sum_loss += lossfn_.loss(*it.second(), ops::dot(w, *it.first()));
    return sum_loss;
  }

  inline void
  worker(tbb::concurrent_bounded_queue<message *> &inq,
         tbb::concurrent_bounded_queue<double> &outq) const
  {
    for (;;) {
      message *px = nullptr;
      inq.pop(px);
      if (!px)
        return;
      const double s = task(*px->w_, *px->d_, px->start_, px->end_);
      outq.push(s);
    }
  }

  inline void
  ensure_pool() const
  {
    if (likely(!thds_.empty()))
      return;
    inqueues_.resize(nthreads_);
    outqueues_.resize(nthreads_);
    for (auto &q : inqueues_)
      q.set_capacity(1);
    for (auto &q : outqueues_)
      q.set_capacity(1);
    messages_.resize(nthreads_);
    for (size_t i = 0; i < nthreads_; i++)
      thds_.emplace_back(&linear_model::worker, this,
          std::ref(inqueues_[i]), std::ref(outqueues_[i]));
  }

  inline void
  shutdown_pool()
  {
    for (auto &q : inqueues_)
      q.push(nullptr);
    for (auto &t : thds_)
      t.join();
  }

  double lambda_;
  standard_vec_t w_;
  LossFunc lossfn_;

  // state for parallel evaluation
  size_t nthreads_;
  mutable std::vector<std::thread> thds_;
  mutable std::vector<tbb::concurrent_bounded_queue<message *>> inqueues_;
  mutable std::vector<tbb::concurrent_bounded_queue<double>> outqueues_;
  mutable std::vector<message> messages_;
};

/**
 * this uses the random projection construction from
 *   Ali Rahimi and Ben Recht.
 *   Random Features for Large-Scale Kernel Machines. NIPS 2007.
 *
 * note this construction only works for translation invariant
 * kernels
 */
template <typename LossFunc, typename Kernel>
class kernelized_linear_model {
public:

  static_assert(Kernel::is_translation_invariant, "xx");
  typedef LossFunc loss_function_type;
  static constexpr const double norm_bound_const = sqrt(2.0);

  kernelized_linear_model(
      double lambda,
      LossFunc lossfn = LossFunc(),
      Kernel kernel = Kernel())
    : underlying_(lambda, lossfn),
      kernel_(kernel)
  { }

  kernelized_linear_model(
      double lambda,
      const standard_vec_t &w,
      LossFunc lossfn = LossFunc(),
      Kernel kernel = Kernel())
    : underlying_(lambda, w, lossfn),
      kernel_(kernel)
  { }

  kernelized_linear_model(
      double lambda,
      standard_vec_t &&w,
      LossFunc lossfn = LossFunc(),
      Kernel kernel = Kernel())
    : underlying_(lambda, std::move(w), lossfn),
      kernel_(kernel)
  { }

  template <typename Generator>
  inline void
  initialize(Generator &prng, size_t xdim, size_t kdim)
  {
    assert(xdim);
    assert(kdim);
    fourier_samples_.resize(kdim);
    for (size_t i = 0; i < kdim; i++)
      fourier_samples_[i] = kernel_.sample_fourier(xdim, prng);
    std::uniform_real_distribution<double> unif(0.0, 2.0 * M_PI);
    b_samples_.resize(kdim);
    for (size_t i = 0; i < kdim; i++)
      b_samples_[i] = unif(prng);
  }

  inline void
  bootstrap(const std::vector<standard_vec_t> &fourier_samples,
            const std::vector<double> &b_samples)
  {
    assert(fourier_samples.size() == b_samples.size());
    fourier_samples_ = fourier_samples;
    b_samples_ = b_samples;
  }

  inline void
  bootstrap(std::vector<standard_vec_t> &&fourier_samples,
            std::vector<double> &&b_samples)
  {
    assert(fourier_samples.size() == b_samples.size());
    fourier_samples_ = std::move(fourier_samples);
    b_samples_ = std::move(b_samples);
  }

  class transformer {
  public:
    transformer(const kernelized_linear_model *impl)
      : impl_(impl) {}

    inline vec_t
    operator()(const vec_t &x) const
    {
      return impl_->transform(x);
    }

    inline size_t postdim() const { return impl_->fourier_samples_.size(); }

  private:
    const kernelized_linear_model *impl_;
  };

  inline transformer
  get_transformer() const
  {
    return transformer(this);
  }

  inline vec_t
  transform(const vec_t &x) const
  {
    vec_t ret;
    standard_vec_t &sret = ret.as_standard_ref();
    sret.resize(fourier_samples_.size());
    for (size_t i = 0; i < fourier_samples_.size(); i++)
      sret[i] = cos(ops::dot(fourier_samples_[i], x) + b_samples_[i]);
    sret *= sqrt(2.0/double(fourier_samples_.size()));
    return ret;
  }

  inline double
  empirical_risk(const dataset &d) const
  {
    dataset transformed(d, get_transformer());
    if (transformed.get_parallel_materialize())
      transformed.materialize();
    return underlying_.empirical_risk(transformed);
  }

  inline standard_vec_t
  grad_empirical_risk(const dataset &d) const
  {
    dataset transformed(d, get_transformer());
    if (transformed.get_parallel_materialize())
      transformed.materialize();
    return underlying_.grad_empirical_risk(transformed);
  }

  inline double
  norm_grad_empirical_risk(const dataset &d) const
  {
    return grad_empirical_risk(d).norm();
  }

  inline standard_vec_t
  predict(const dataset &d) const
  {
    dataset transformed(d, get_transformer());
    if (transformed.get_parallel_materialize())
      transformed.materialize();
    return underlying_.predict(transformed);
  }

  inline dataset
  transform(const dataset &d) const
  {
    return dataset(d, get_transformer());
  }

  inline double get_lambda() const { return underlying_.get_lambda(); }
  inline standard_vec_t & weightvec() { return underlying_.weightvec(); }
  inline const standard_vec_t & weightvec() const { return underlying_.weightvec(); }
  inline const LossFunc & get_lossfn() const { return underlying_.get_lossfn(); }
  inline const Kernel & get_kernel() const { return kernel_; }

  inline kernelized_linear_model<LossFunc, Kernel>
  buildfrom(const standard_vec_t &w) const
  {
    kernelized_linear_model<LossFunc, Kernel> ret(
        underlying_.get_lambda(), w, underlying_.get_lossfn(), kernel_);
    ret.bootstrap(fourier_samples_, b_samples_);
    return ret;
  }

  inline kernelized_linear_model<LossFunc, Kernel>
  buildfrom(standard_vec_t &&w) const
  {
    kernelized_linear_model<LossFunc, Kernel> ret(
        underlying_.get_lambda(), std::move(w), underlying_.get_lossfn(), kernel_);
    ret.bootstrap(fourier_samples_, b_samples_);
    return ret;
  }

  inline std::map<std::string, std::string>
  mapconfig() const
  {
    std::map<std::string, std::string> m = underlying_.mapconfig();
    m["model_type"] = "kernelized_linear";
    return m;
  }

private:
  linear_model<LossFunc> underlying_;
  Kernel kernel_;

  // the randomized basis vectors are
  // phi_i(x) = cos(<fourier_samples_[i],x> + b_samples_[i])
  std::vector<standard_vec_t> fourier_samples_; // number of reduced directions
  std::vector<double> b_samples_; // same size as fourier_samples_
};

template <typename Model>
struct model_history {
  model_history(size_t iteration, size_t runtime_usec,
                const Model &model)
    : iteration_(iteration), runtime_usec_(runtime_usec),
      model_(model) {}
  model_history(size_t iteration, size_t runtime_usec,
                Model &&model)
    : iteration_(iteration), runtime_usec_(runtime_usec),
      model_(std::move(model)) {}
  size_t iteration_; // 1-based
  size_t runtime_usec_; // the amount of time relative to the previous history entry
  Model model_;
};

} // namespace model
