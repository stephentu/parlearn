#pragma once

#include <vector>
#include <memory>
#include <string>

#include <macros.hh>
#include <vec.hh>
#include <loss_functions.hh>
#include <dataset.hh>

namespace classifier {

template <typename Model>
class clf_iface {
public:
  virtual ~clf_iface() {}
  virtual void fit(const dataset &d, bool keep_histories=false) = 0;
  virtual const Model & get_model() const = 0;
  // [0, get_nhistory_samples())
  virtual model::model_history<Model>
    history(size_t sample_id) = 0;
  virtual size_t get_nhistory_samples() const = 0;
  virtual standard_vec_t predict(const dataset &d) const = 0;
  virtual size_t get_nrounds() const = 0;
  virtual clf_iface<Model> *clone() const = 0;
  virtual std::string name() const = 0;
  virtual std::map<std::string, std::string> mapconfig() const = 0;
};

template <typename Impl>
class clf_delegator : public clf_iface<typename Impl::model_type> {
public:
  clf_delegator(const Impl &impl) : impl_(impl) {}
  clf_delegator(Impl &&impl) : impl_(std::move(impl)) {}
  void fit(const dataset &d, bool keep_histories) OVERRIDE
    { impl_.fit(d, keep_histories); }
  const typename Impl::model_type & get_model() const OVERRIDE { return impl_.get_model(); }
  model::model_history<typename Impl::model_type>
    history(size_t round) OVERRIDE { return impl_.history(round); }
  size_t get_nhistory_samples() const OVERRIDE { return impl_.get_nhistory_samples(); }
  standard_vec_t predict(const dataset &d) const OVERRIDE { return impl_.get_model().predict(d); }
  size_t get_nrounds() const OVERRIDE { return impl_.get_nrounds(); }
  clf_iface<typename Impl::model_type> *clone() const OVERRIDE { return new clf_delegator(impl_); }
  std::string name() const OVERRIDE { return impl_.name(); }
  std::map<std::string, std::string> mapconfig() const OVERRIDE { return impl_.mapconfig(); }
private:
  Impl impl_;
};

template <typename Model, typename Generator>
class base_iterative_clf {
public:
  base_iterative_clf(const Model &model,
                     size_t nrounds,
                     const std::shared_ptr<Generator> &prng,
                     bool verbose)
    : model_(model),
      nrounds_(nrounds),
      training_sz_(0),
      prng_(prng),
      verbose_(verbose)
  {
    assert(nrounds > 0);
  }

  base_iterative_clf(const base_iterative_clf &clf)
    : model_(clf.model_),
      nrounds_(clf.nrounds_),
      training_sz_(clf.training_sz_),
      prng_(new Generator(std::uniform_int_distribution<unsigned>()(*clf.prng_))),
      verbose_(clf.verbose_)
  {

  }

  base_iterative_clf &operator=(const base_iterative_clf &clf) = delete;

  inline Model & get_model() { return model_; }
  inline const Model & get_model() const { return model_; }
  inline size_t get_nrounds() const { return nrounds_; }
  inline size_t get_training_sz() const { return training_sz_; }

  // [iteration ID (1-based), model]
  virtual model::model_history<Model>
  history(size_t i)
  {
    ALWAYS_ASSERT(i < get_nhistory_samples());
    return model::model_history<Model>(
        this->w_history_[i].iteration_,
        this->w_history_[i].runtime_usec_,
        std::move(model_.buildfrom(this->w_history_[i].w_)));
  }

  virtual size_t
  get_nhistory_samples() const
  {
    return this->w_history_.size();
  }

  virtual std::string name() const = 0;

  virtual std::map<std::string, std::string>
  mapconfig() const
  {
    std::map<std::string, std::string> m = this->model_.mapconfig();
    m["clf_name"]        = name();
    m["clf_nrounds"]     = std::to_string(nrounds_);
    m["clf_training_sz"] = std::to_string(training_sz_);
    return m;
  }

  inline std::string
  jsonconfig() const
  {
    return util::smap_to_json(mapconfig());
  }

protected:
  Model model_;
  size_t nrounds_;
  size_t training_sz_;
  std::shared_ptr<Generator> prng_;
  bool verbose_;
  struct state_entry {
    size_t iteration_;
    size_t runtime_usec_;
    standard_vec_t w_; // all models have a w_ in common

    state_entry() = default;
    state_entry(size_t iteration, size_t runtime_usec, const standard_vec_t &w)
      : iteration_(iteration),
        runtime_usec_(runtime_usec),
        w_(w)
    {}
  };
  std::vector<state_entry> w_history_;
};

} // namespace classifier
