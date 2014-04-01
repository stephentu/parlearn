#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <type_traits>
#include <vec.hh>
#include <util.hh>
#include <macros.hh>

class dataset {
public:

  class storage_iface {
  public:
    virtual const vec_t &
      get_x(size_t idx) const = 0;
    virtual const double &
      get_y(size_t idx) const = 0;
    virtual std::pair<const vec_t *, double>
    get(size_t idx) const
    {
      return std::make_pair(&get_x(idx), get_y(idx));
    }
    virtual const standard_vec_t & get_raw_y() const = 0;
    virtual std::pair<size_t, size_t>
      x_shape() const = 0;
    virtual bool can_be_materialized() const = 0;
  };

  // XXX: there is a better way to do this in C++11 (can
  // store function pointer for members), but laziness
  struct storage_iface_x_extractor {
    typedef const vec_t & return_type;
    inline return_type
    operator()(const storage_iface &iface, size_t idx) const
    {
      return iface.get_x(idx);
    }
  };
  struct storage_iface_y_extractor {
    typedef const double & return_type;
    inline return_type
    operator()(const storage_iface &iface, size_t idx) const
    {
      return iface.get_y(idx);
    }
  };

  template <typename T>
  struct itertype {
    typedef
      typename std::remove_const<
        typename std::remove_reference<T>::type
      >::type type;
  };

  template <typename Extractor>
  class const_iterator_impl :
    public std::iterator<
      std::forward_iterator_tag,
      const typename itertype<typename Extractor::return_type>::type> {
    friend class dataset;
  public:
    const_iterator_impl() = default;
    const_iterator_impl(const const_iterator_impl &) = default;

    typedef const typename itertype<typename Extractor::return_type>::type & reference_type;
    typedef const typename itertype<typename Extractor::return_type>::type * pointer_type;

    inline reference_type
    operator*() const
    {
      return Extractor()(*d_->storage_, p_ ? (*p_)[idx_] : idx_);
    }

    inline pointer_type
    operator->() const
    {
      return &Extractor()(*d_->storage_, p_ ? (*p_)[idx_] : idx_);
    }

    inline bool
    operator==(const const_iterator_impl &o) const
    {
      // XXX: doesn't check to see if datasets are the same, or if they are
      // over the same permutation
      return idx_ == o.idx_;
    }

    inline bool
    operator!=(const const_iterator_impl &o) const
    {
      return !operator==(o);
    }

    inline const_iterator_impl
    operator+(ptrdiff_t off) const
    {
      return const_iterator_impl(d_, p_, idx_ + off);
    }

    inline ptrdiff_t
    operator-(const const_iterator_impl &o) const
    {
      return ptrdiff_t(idx_) - ptrdiff_t(o.idx_);
    }

    inline const_iterator_impl &
    operator++()
    {
      ++idx_;
      return *this;
    }

    inline const_iterator_impl
    operator++(int)
    {
      const_iterator_impl cur = *this;
      ++(*this);
      return cur;
    }

  protected:
    const_iterator_impl(
        const dataset *d,
        const std::vector<size_t> *p,
        size_t idx)
      : d_(d), p_(p), idx_(idx) {}

    const dataset *d_;
    const std::vector<size_t> *p_;
    size_t idx_;
  };

  /**
   * not really a valid iterator in the STL sense, but still useful
   */
  template <typename ForwardIterA, typename ForwardIterB>
  class zip_iterator {
  public:
    zip_iterator() = default;
    zip_iterator(ForwardIterA a, ForwardIterB b)
      : a_(a), b_(b) {}

    inline ForwardIterA & first() { return a_; }
    inline ForwardIterB & second() { return b_; }

    inline const ForwardIterA & first() const { return a_; }
    inline const ForwardIterB & second() const { return b_; }

    inline bool
    operator==(const zip_iterator &o) const
    {
      return a_ == o.a_ && b_ == o.b_;
    }

    inline bool
    operator!=(const zip_iterator &o) const
    {
      return !operator==(o);
    }

    inline zip_iterator
    operator+(ptrdiff_t off) const
    {
      return zip_iterator(a_ + off, b_ + off);
    }

    inline ptrdiff_t
    operator-(const zip_iterator &o) const
    {
      return a_ - o.a_;
    }

    inline zip_iterator &
    operator++()
    {
      ++a_; ++b_;
      return *this;
    }

    inline zip_iterator
    operator++(int)
    {
      zip_iterator cur = *this;
      ++(*this);
      return cur;
    }

  private:
    ForwardIterA a_;
    ForwardIterB b_;
  };

  class vector_storage : public storage_iface {
  public:
    vector_storage(const std::vector<vec_t> &x,
                   const standard_vec_t &y)
      : x_(x), y_(y)
    {
      assert(x_.size() == y_.size());
    }
    vector_storage(std::vector<vec_t> &&x,
                   standard_vec_t &&y)
      : x_(std::move(x)), y_(std::move(y))
    {
      assert(x_.size() == y_.size());
    }
    const vec_t &
    get_x(size_t idx) const OVERRIDE
    {
      return x_[idx];
    }
    const double &
    get_y(size_t idx) const OVERRIDE
    {
      return y_[idx];
    }
    std::pair<size_t, size_t>
    x_shape() const OVERRIDE
    {
      size_t nfeatures = 0;
      for (auto &x : x_)
        nfeatures = std::max(nfeatures, x.highest_nonzero_dim());
      return std::make_pair(x_.size(), nfeatures);
    }
    const standard_vec_t &
    get_raw_y() const OVERRIDE
    {
      return y_;
    }
    bool
    can_be_materialized() const OVERRIDE
    {
      return false;
    }
  private:
    std::vector<vec_t> x_;
    standard_vec_t y_;
  };

  template <typename Transformer>
  class transforming_storage : public storage_iface {
  public:
    transforming_storage(
        const std::shared_ptr<storage_iface> &impl,
        Transformer trfm)
      : impl_(impl), trfm_(trfm) {}
    const vec_t &
    get_x(size_t idx) const OVERRIDE
    {
      return const_cast<transforming_storage *>(this)->sync(idx);
    }
    const double &
    get_y(size_t idx) const OVERRIDE
    {
      return impl_->get_y(idx);
    }
    std::pair<size_t, size_t>
    x_shape() const OVERRIDE
    {
      const auto underlying_shape = impl_->x_shape();
      return std::make_pair(underlying_shape.first, trfm_.postdim());
    }
    const standard_vec_t &
    get_raw_y() const OVERRIDE
    {
      return impl_->get_raw_y();
    }
    bool
    can_be_materialized() const OVERRIDE
    {
      return true;
    }
  private:
    inline const vec_t &
    sync(size_t idx)
    {
      return (v_.my() = trfm_(impl_->get_x(idx)));
    }
    std::shared_ptr<storage_iface> impl_;
    Transformer trfm_;
    util::percore<vec_t> v_;
  };

  dataset(const std::vector<vec_t> &x, const standard_vec_t &y)
    : storage_(new vector_storage(x, y)),
      parallel_materialize_(false)
  {
    initshape();
  }

  dataset(std::vector<vec_t> &&x, standard_vec_t &&y)
    : storage_(new vector_storage(std::move(x), std::move(y))),
      parallel_materialize_(false)
  {
    initshape();
  }

  template <typename Transformer>
  dataset(const dataset &that, Transformer trfm)
    : storage_(new transforming_storage<Transformer>(that.storage_, trfm)),
      parallel_materialize_(that.parallel_materialize_)
  {
    initshape();
  }

  void
  set_parallel_materialize(bool parallel_materialize)
  {
    parallel_materialize_ = parallel_materialize;
  }

  inline bool get_parallel_materialize() const { return parallel_materialize_; }

  inline const vec_t &
  get_x(size_t idx) const
  {
    return storage_->get_x(idx);
  }

  inline const standard_vec_t &
  get_y() const
  {
    return storage_->get_raw_y();
  }

  typedef const_iterator_impl<storage_iface_x_extractor> x_const_iterator;
  typedef const_iterator_impl<storage_iface_y_extractor> y_const_iterator;
  typedef zip_iterator<x_const_iterator, y_const_iterator> const_iterator;

  inline x_const_iterator
  x_begin() const
  {
    return x_const_iterator(this, nullptr, 0);
  }

  inline x_const_iterator
  x_end() const
  {
    return x_const_iterator(this, nullptr, x_shape_.first);
  }

  inline y_const_iterator
  y_begin() const
  {
    return y_const_iterator(this, nullptr, 0);
  }

  inline y_const_iterator
  y_end() const
  {
    return y_const_iterator(this, nullptr, x_shape_.first);
  }

  inline const_iterator
  begin() const
  {
    return const_iterator(x_begin(), y_begin());
  }

  inline const_iterator
  end() const
  {
    return const_iterator(x_end(), y_end());
  }

  inline std::pair<size_t, size_t>
  get_x_shape() const
  {
    return x_shape_;
  }

  double
  max_x_norm() const
  {
    double best = 0.0;
    const auto end = x_end();
    for (auto it = x_begin(); it != end; ++it)
      best = std::max(best, (*it).norm());
    return best;
  }

  class permutation {
    friend class dataset;
  public:
    inline x_const_iterator
    x_begin() const
    {
      return x_const_iterator(d_, &pi_, 0);
    }
    inline x_const_iterator
    x_end() const
    {
      return x_const_iterator(d_, &pi_, d_->x_shape_.first);
    }
    inline y_const_iterator
    y_begin() const
    {
      return y_const_iterator(d_, &pi_, 0);
    }
    inline y_const_iterator
    y_end() const
    {
      return y_const_iterator(d_, &pi_, d_->x_shape_.first);
    }
    inline const_iterator
    begin() const
    {
      return const_iterator(x_begin(), y_begin());
    }
    inline const_iterator
    end() const
    {
      return const_iterator(x_end(), y_end());
    }
  private:
    permutation(const dataset *d,
                const std::vector<size_t> &pi)
      : d_(d), pi_(pi) {}
    permutation(const dataset *d,
                std::vector<size_t> &&pi)
      : d_(d), pi_(std::move(pi)) {}
    const dataset *d_;
    std::vector<size_t> pi_;
  };

  template <typename Generator>
  inline permutation
  permute(Generator &g) const
  {
    std::vector<size_t> pi = util::range(x_shape_.first);
    for (size_t i = pi.size() - 1; i >= 1; i--) {
      std::uniform_int_distribution<> dist(0, i);
      const size_t j = dist(g);
      std::swap(pi[j], pi[i]);
    }
    return permutation(this, std::move(pi));
  }

  void
  materialize()
  {
    if (!storage_->can_be_materialized())
      return;
    if (!parallel_materialize_ || !do_parallel_materialize()) {
      std::vector<vec_t> x(x_begin(), x_end());
      standard_vec_t y(get_y());
      assert(x.size() == x_shape_.first);
      assert(y.size() == x_shape_.first);
      storage_.reset(new vector_storage(std::move(x), std::move(y)));
    }
  }

private:

  bool do_parallel_materialize();

  inline void
  initshape()
  {
    x_shape_ = storage_->x_shape();
  }

  std::shared_ptr<storage_iface> storage_;
  std::pair<size_t, size_t> x_shape_;
  bool parallel_materialize_;
};
