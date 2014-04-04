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

namespace opt {

/**
 * For testing/debugging purposes
 */
template <typename Model, typename Generator>
class gd : public classifier::base_iterative_clf<Model, Generator> {
public:

  typedef Model model_type;
  typedef Generator generator_type;

  gd(const Model &model,
     size_t nrounds,
     const std::shared_ptr<Generator> &prng,
     size_t t_offset = 0,
     double c0 = 1.0,
     bool verbose = false)
    : classifier::base_iterative_clf<Model, Generator>(model, nrounds, prng, verbose),
      t_offset_(t_offset),
      c0_(c0)
  {
    ALWAYS_ASSERT(c0_ > 0.0);
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

    this->w_history_.clear();
    if (keep_histories)
      this->w_history_.reserve(this->nrounds_);
    this->model_.weightvec().resize(shape.second);

    standard_vec_t accum(shape.second);
    for (size_t round = 0; round < this->nrounds_; round++) {
      const size_t t_eff = (1+round) + t_offset_;
      const double eta_t = c0_ / (this->model_.get_lambda() * t_eff);
      accum.reset();

      const auto it_end = transformed.end();
      for (auto it = transformed.begin(); it != it_end; ++it) {
        const auto &x = *it.first();
        const double dloss = this->model_.get_lossfn().dloss(
            *it.second(), ops::dot(this->model_.weightvec(), x));
        const auto inner_it_end = x.end();
        for (auto inner_it = x.begin();
            inner_it != inner_it_end; ++inner_it) {
          const size_t feature_idx = inner_it.tell();
          accum[feature_idx] += (*inner_it) * dloss;
        }
      }

      accum *= (eta_t / double(this->training_sz_));
      this->model_.weightvec() *= (1.0 - eta_t * this->model_.get_lambda());
      this->model_.weightvec() -= accum;

      if (this->verbose_) {
        std::cerr << "[INFO] finished round " << (round+1) << std::endl;
        std::cerr << "[INFO] current risk: "
                  << this->model_.empirical_risk(transformed) << std::endl;
        std::cerr << "[INFO] step size: " << eta_t << std::endl;
      }
    }
  }

  inline size_t get_t_offset() const { return t_offset_; }
  inline double get_c0() const { return c0_; }

  std::string name() const OVERRIDE { return "gd"; }

  std::map<std::string, std::string>
  mapconfig() const OVERRIDE
  {
    std::map<std::string, std::string> m =
      classifier::base_iterative_clf<Model, Generator>::mapconfig();
    m["clf_name"]     = name();
    m["clf_t_offset"] = std::to_string(t_offset_);
    m["clf_c0"]       = std::to_string(c0_);
    return m;
  }

private:
  size_t t_offset_;
  double c0_;
};

} // namespace opt
