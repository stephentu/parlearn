#pragma once

#include <cassert>
#include <vector>
#include <random>
#include <memory>
#include <string>
#include <cmath>

#include <macros.hh>
#include <vec.hh>
#include <pretty_printers.hh>
#include <loss_functions.hh>
#include <dataset.hh>
#include <timer.hh>
#include <model.hh>
#include <classifier.hh>
#include <lvec.hh>
#include <task_executor.hh>

namespace opt {

template <typename Model, typename Generator>
class parsgd : public classifier::base_iterative_clf<Model, Generator> {
public:

  typedef Model model_type;
  typedef Generator generator_type;

  parsgd(const Model &model,
         size_t nrounds,
         const std::shared_ptr<Generator> &prng,
         size_t nworkers,
         bool do_locking,
         size_t t_offset = 0,
         double c0 = 1.0,
         bool verbose = false)
    : classifier::base_iterative_clf<Model, Generator>(model, nrounds, prng, verbose),
      t_offset_(t_offset),
      c0_(c0),
      nworkers_(nworkers),
      do_locking_(do_locking)
  {
    ALWAYS_ASSERT(c0_ > 0.0);
    ALWAYS_ASSERT(nworkers_ > 0);
  }

  void
  fit(const dataset& d, bool keep_histories=false)
  {
    dataset transformed(this->model_.transform(d));
    if (this->verbose_)
      std::cerr << "[INFO] fitting x_shape: "
                << transformed.get_x_shape() << std::endl;
    timer tt;
    transformed.materialize();
    if (this->verbose_) {
      std::cerr << "[INFO] materializing took " << tt.lap_ms() << " ms" << std::endl;
      std::cerr << "[INFO] max transformed norm is " << transformed.max_x_norm() << std::endl;
    }

    const auto shape = transformed.get_x_shape();
    this->training_sz_ = shape.first;
    const auto feature_counts = transformed.feature_counts();

    //if (this->verbose_) {
    //  for (size_t i = 0; i < feature_counts.size(); i++)
    //    if (!feature_counts[i])
    //      std::cerr << "[WARN] feature idx " << i << " is never used!" << std::endl;
    //}

    this->state_.reset(new standard_lvec<double>(shape.second));
    this->w_history_.clear();
    if (keep_histories)
      this->w_history_.reserve(this->nrounds_);

    // setup executors
    std::vector< std::unique_ptr<task_executor_thread<bool>> > workers;
    const size_t actual_nworkers =
      (this->training_sz_ < nworkers_) ? 1 : nworkers_;
    if (this->verbose_) {
      std::cerr << "[INFO] keep_histories: " << keep_histories << std::endl;
      std::cerr << "[INFO] actual_nworkers: " << actual_nworkers << std::endl;
      std::cerr << "[INFO] starting eta_t: "
                << c0_ / (this->model_.get_lambda() * (1 + this->t_offset_))
                << std::endl;
    }
    for (size_t i = 0; i < actual_nworkers; i++)
      workers.emplace_back(new task_executor_thread<bool>);
    const size_t nelems_per_worker =
      this->training_sz_ / actual_nworkers;
    tt.lap();
    std::vector<std::future<bool>> futures;
    timer tt1;
    for (size_t round = 0; round < this->nrounds_; round++) {
      const auto permutation = transformed.permute(*this->prng_);
      const auto it_end = permutation.end();
      const auto it_beg = permutation.begin();

      // Uncomment (and comment out above) to remove randomness each round
      // (testing purposes)
      /**
      const auto it_end = transformed.end();
      const auto it_beg = transformed.begin();
      */

      tt1.lap();
      for (size_t i = 0; i < actual_nworkers; i++)
        futures.emplace_back(
          workers[i]->enq(
            std::bind(
              do_locking_ ? &parsgd::work<true> : &parsgd::work<false>,
              this,
              round+1,
              this->training_sz_,
              std::ref(feature_counts),
              it_beg + i*nelems_per_worker,
              ((i+1)==actual_nworkers) ?
                it_end : (it_beg + (i+1)*nelems_per_worker))));
      for (auto &f : futures)
        f.wait();
      futures.clear();
      if (keep_histories) {
        state_->unsafesnapshot(this->model_.weightvec());
        this->w_history_.emplace_back(
            round + 1, tt.elapsed_usec(), this->model_.weightvec());
      }

      if (this->verbose_) {
        std::cerr << "[INFO] finished round " << (round+1) << " in "
                  << tt1.lap_ms() << " ms" << std::endl;
        state_->unsafesnapshot(this->model_.weightvec());
        std::cerr << "[INFO] current risk: "
                  << this->model_.empirical_risk(transformed) << std::endl;
      }
    }
    state_->unsafesnapshot(this->model_.weightvec());
    for (auto &w : workers)
      w->shutdown();
    ALWAYS_ASSERT( this->model_.weightvec().size() == shape.second );
  }

  inline size_t get_t_offset() const { return t_offset_; }
  inline double get_c0() const { return c0_; }
  inline size_t get_nworkers() const { return nworkers_; }
  inline bool get_do_locking() const { return do_locking_; }

  std::string name() const OVERRIDE { return "parsgd"; }

  std::map<std::string, std::string>
  mapconfig() const OVERRIDE
  {
    std::map<std::string, std::string> m =
      classifier::base_iterative_clf<Model, Generator>::mapconfig();
    m["clf_name"]       = name();
    m["clf_t_offset"]   = std::to_string(t_offset_);
    m["clf_c0"]         = std::to_string(c0_);
    m["clf_nworkers"]   = std::to_string(nworkers_);
    m["clf_do_locking"] = std::to_string(do_locking_);
    return m;
  }

private:

  template <bool DoLocking>
  static inline double
  dot(const vec_t &x, standard_lvec<double> &b)
  {
    double s = 0.0;
    const auto inner_it_end = x.end();
    for (auto inner_it = x.begin();
         inner_it != inner_it_end; ++inner_it) {
      const size_t feature_idx = inner_it.tell();
      if (DoLocking)
        s += (*inner_it) * b.lockandread(feature_idx);
      else
        s += (*inner_it) * b.unsaferead(feature_idx);
    }
    return s;
  }

  template <bool DoLocking>
  bool
  work(size_t round,
       size_t dataset_size,
       const std::vector<size_t> &feature_counts,
       dataset::const_iterator begin,
       dataset::const_iterator end)
  {
    const double dataset_sizef = double(dataset_size);
    size_t i = 1;
    for (auto it = begin; it != end; ++it, ++i) {
      const size_t t_eff = (round-1)*dataset_size + i + t_offset_;
      const double eta_t = c0_ / (this->model_.get_lambda() * t_eff);
      const auto &x = *it.first();
      const double dloss = this->model_.get_lossfn().dloss(
          *it.second(), dot<DoLocking>(x, *state_.get()));
      const auto inner_it_end = x.end();
      for (auto inner_it = x.begin();
           inner_it != inner_it_end; ++inner_it) {
        const size_t feature_idx = inner_it.tell();
        const double w_old = state_->unsaferead(feature_idx);
        assert(feature_counts[feature_idx]);
        const double w_new =
          (1.0 - eta_t * this->model_.get_lambda() * dataset_sizef /
           double(feature_counts[feature_idx])) * w_old
          - eta_t * dloss * (*inner_it);
        if (DoLocking)
          state_->writeandunlock(feature_idx, w_new);
        else
          state_->unsafewrite(feature_idx, w_new);
      }
    }
    return false;
  }

  size_t t_offset_;
  double c0_;
  size_t nworkers_;
  bool do_locking_;
  std::unique_ptr<standard_lvec<double>> state_;
};

} // namespace opt
