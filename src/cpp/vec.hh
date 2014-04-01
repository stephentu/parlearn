#pragma once

#include <algorithm>
#include <vector>
#include <utility>
#include <iterator>
#include <cmath>
#include <cstddef>

#include <pretty_printers.hh>
#include <macros.hh>
#include <util.hh>

template <typename T> class vec;
template <typename T> class standard_vec;
template <typename T> class sparse_vec;

template <typename T>
class vec_const_iterator :
  public std::iterator<std::forward_iterator_tag, const T> {
  friend class vec<T>;
public:

  vec_const_iterator() = default;
  vec_const_iterator(const vec_const_iterator &) = default;
  vec_const_iterator &operator=(const vec_const_iterator &) = default;
  vec_const_iterator(vec_const_iterator &&) = default;

  inline const T &
  operator*() const
  {
    return std_ ? *std_iter_ : sparse_iter_->second;
  }

  inline const T *
  operator->() const
  {
    return std_ ? &(*std_iter_) : &sparse_iter_->second;
  }

  inline bool
  operator==(const vec_const_iterator &that) const
  {
    return std_ ?
      (std_iter_ == that.std_iter_) :
      (sparse_iter_ == that.sparse_iter_);
  }

  inline bool
  operator!=(const vec_const_iterator &that) const
  {
    return !operator==(that);
  }

  inline vec_const_iterator &
  operator++()
  {
    if (std_)
      ++std_iter_;
    else
      ++sparse_iter_;
    return *this;
  }

  inline vec_const_iterator
  operator++(int)
  {
    vec_const_iterator cur(*this);
    ++(*this);
    return cur;
  }

  inline size_t
  tell() const
  {
    return std_ ?
      (std_iter_ - std_iter_begin_) :
      sparse_iter_->first;
  }

protected:
  vec_const_iterator(const vec<T> &v, bool begin);

private:
  bool std_;
  typename std::vector<T>::const_iterator std_iter_begin_;
  typename std::vector<T>::const_iterator std_iter_;
  typename std::vector<std::pair<size_t, T>>::const_iterator sparse_iter_;
};

template <typename T>
class vec {
  friend class vec_const_iterator<T>;
public:
  static_assert(std::is_floating_point<T>::value, "need FP type");

  struct std_tag_t {};
  struct sparse_tag_t {};
  enum class tag : uint8_t { STD, SPARSE };

  vec() : tag_(tag::STD) {}
  vec(const vec &) = default;
  vec &operator=(const vec &) = default;
  vec(vec &&) = default;

  vec(std_tag_t)
    : tag_(tag::STD) {}
  vec(sparse_tag_t)
    : tag_(tag::SPARSE) {}

  template <typename U>
  vec(std_tag_t, const std::vector<U> &std_repr)
    : tag_(tag::STD), std_repr_(std_repr), sparse_repr_() {}
  vec(std_tag_t, std::vector<T> &&std_repr)
    : tag_(tag::STD), std_repr_(std::move(std_repr)), sparse_repr_() {}

  template <typename U>
  vec(sparse_tag_t, const std::vector<std::pair<size_t, U>> &sparse_repr)
    : tag_(tag::SPARSE), std_repr_(), sparse_repr_(sparse_repr) {}
  vec(sparse_tag_t, std::vector<std::pair<size_t, T>> &&sparse_repr)
    : tag_(tag::SPARSE), std_repr_(), sparse_repr_(std::move(sparse_repr)) {}

  // no checking guaranteed
  inline standard_vec<T> * as_standard_ptr();
  inline const standard_vec<T> * as_standard_ptr() const;
  inline sparse_vec<T> * as_sparse_ptr();
  inline const sparse_vec<T> * as_sparse_ptr() const;

  inline standard_vec<T> & as_standard_ref();
  inline const standard_vec<T> & as_standard_ref() const;
  inline sparse_vec<T> & as_sparse_ref();
  inline const sparse_vec<T> & as_sparse_ref() const;

  // ensures the vector is at least (i+1) dimensions first
  inline T &ensureref(size_t i);
  inline T norm() const; // l2 norm

  inline T sum() const;
  inline void reserve(size_t n);
  inline size_t highest_nonzero_dim() const;
  inline size_t nnz() const;

  typedef vec_const_iterator<T> const_iterator;

  inline const_iterator
  begin() const
  {
    return const_iterator(*this, true);
  }

  inline const_iterator
  end() const
  {
    return const_iterator(*this, false);
  }

  inline tag get_tag() const { return tag_; }
  inline bool is_standard() const { return tag_ == tag::STD; }
  inline bool is_sparse() const { return tag_ == tag::SPARSE; }

protected:
  tag tag_;
  std::vector<T> std_repr_;
  std::vector<std::pair<size_t, T>> sparse_repr_;
};

template <typename T>
vec_const_iterator<T>::vec_const_iterator(const vec<T> &v, bool begin)
  : std_(v.is_standard())
{
  if (std_) {
    std_iter_begin_ = v.std_repr_.begin();
    std_iter_ = begin ? v.std_repr_.begin() : v.std_repr_.end();
  } else {
    sparse_iter_ = begin ? v.sparse_repr_.begin() : v.sparse_repr_.end();
  }
}

// most commonly used vector type
typedef vec<double> vec_t;

template <typename T>
class standard_vec : public vec<T> {
public:

  typedef T value_type;

  standard_vec() : vec<T>(typename vec<T>::std_tag_t()) {}
  standard_vec(size_t n)
    : vec<T>(typename vec<T>::std_tag_t(), std::vector<T>(n)) {}

  template <typename U>
  standard_vec(std::initializer_list<U> elems)
    : vec<T>(typename vec<T>::std_tag_t(),
             std::vector<T>(elems.begin(), elems.end())) {}

  standard_vec(const standard_vec &) = default;
  standard_vec &operator=(const standard_vec &) = default;
  standard_vec(standard_vec &&) = default;

  template <typename U>
  standard_vec(const std::vector<U> &v)
    : vec<T>(typename vec<T>::std_tag_t(), v) {}

  standard_vec(std::vector<T> &&v)
    : vec<T>(typename vec<T>::std_tag_t(), std::move(v)) {}

          /** vec api **/

  inline T &
  ensureref(size_t i)
  {
    assert(this->tag_ == vec<T>::tag::STD);
    if (this->std_repr_.size() <= i)
      this->std_repr_.resize(i + 1);
    return this->std_repr_[i];
  }

  inline T norm() const;

  inline T
  infnorm() const
  {
    return map([](T t) { return fabs(t); }).max();
  }

  inline T
  max() const
  {
    assert(this->tag_ == vec<T>::tag::STD);
    T best = this->std_repr_[0];
    for (auto it = this->std_repr_.begin() + 1;
         it != this->std_repr_.end();
         ++it) {
      best = std::max(best, *it);
    }
    return best;
  }

  template <typename Fn>
  inline standard_vec<T>
  map(Fn f) const
  {
    assert(this->tag_ == vec<T>::tag::STD);
    std::vector<T> ret;
    ret.reserve(this->std_repr_.size());
    for (auto p : this->std_repr_)
      ret.push_back(f(p));
    return standard_vec<T>(std::move(ret));
  }

  inline T
  sum() const
  {
    assert(this->tag_ == vec<T>::tag::STD);
    T accum = T();
    for (auto e : this->std_repr_)
      accum += e;
    return accum;
  }

  inline T
  mean() const
  {
    assert(this->tag_ == vec<T>::tag::STD);
    return sum() / T(size());
  }

  inline T
  std(size_t dof=0) const
  {
    return sqrt(var(dof));
  }

  inline T
  var(size_t dof=0) const
  {
    const T mu = mean();
    T s1 = T();
    for (auto x : this->std_repr_)
      s1 += (x - mu) * (x - mu);
    return s1 / T(size() - dof);
  }

  inline standard_vec<T>
  cumsum() const
  {
    assert(this->tag_ == vec<T>::tag::STD);
    standard_vec<T> ret(size());
    T accum = T();
    for (size_t i = 0; i < size(); i++) {
      ret[i] = (*this)[i] + accum;
      accum += (*this)[i];
    }
    return ret;
  }

  template <typename Predicate>
  inline size_t
  count(Predicate p) const
  {
    assert(this->tag_ == vec<T>::tag::STD);
    size_t count = 0;
    for (auto x : this->std_repr_)
      if (p(x))
        count++;
    return count;
  }

  inline void
  reserve(size_t n)
  {
    assert(this->tag_ == vec<T>::tag::STD);
    this->std_repr_.reserve(n);
  }

  inline void
  resize(size_t n)
  {
    assert(this->tag_ == vec<T>::tag::STD);
    this->std_repr_.resize(n);
  }

  inline void
  zero()
  {
    assert(this->tag_ == vec<T>::tag::STD);
    for (auto &e : this->std_repr_)
      e = T();
  }

  inline size_t
  highest_nonzero_dim() const
  {
    assert(this->tag_ == vec<T>::tag::STD);
    return this->std_repr_.size();
  }

  inline size_t
  nnz() const
  {
    return size();
  }

          /** specific api **/

  inline size_t
  size() const
  {
    assert(this->tag_ == vec<T>::tag::STD);
    return this->std_repr_.size();
  }

  inline T &
  operator[](size_t i)
  {
    assert(this->tag_ == vec<T>::tag::STD);
    return this->std_repr_[i];
  }

  inline const T &
  operator[](size_t i) const
  {
    assert(this->tag_ == vec<T>::tag::STD);
    return this->std_repr_[i];
  }

  inline const std::vector<T> &
  data() const
  {
    assert(this->tag_ == vec<T>::tag::STD);
    return this->std_repr_;
  }

  inline std::vector<T> &
  data()
  {
    assert(this->tag_ == vec<T>::tag::STD);
    return this->std_repr_;
  }

  inline standard_vec
  sign() const
  {
    assert(this->tag_ == vec<T>::tag::STD);
    standard_vec ret;
    ret.resize(size());
    const size_t d = size();
    for (size_t i = 0; i < d; i++)
      ret[i] = util::sign(this->std_repr_[i]);
    return ret;
  }

  template <typename U>
  inline standard_vec &
  operator+=(const standard_vec<U> &b)
  {
    assert(this->tag_ == vec<T>::tag::STD);
    assert(size() == b.size());
    const size_t d = size();
    for (size_t i = 0; i < d; i++)
      this->std_repr_[i] += b[i];
    return *this;
  }

  template <typename U>
  inline standard_vec &
  operator+=(const sparse_vec<U> &b)
  {
    assert(this->tag_ == vec<T>::tag::STD);
    for (auto &p : b.data())
      this->std_repr_[p.first] += p.second;
    return *this;
  }

  template <typename U>
  inline standard_vec &
  operator-=(const standard_vec<U> &b)
  {
    assert(this->tag_ == vec<T>::tag::STD);
    assert(size() == b.size());
    const size_t d = size();
    for (size_t i = 0; i < d; i++)
      this->std_repr_[i] -= b[i];
    return *this;
  }

  template <typename U>
  inline standard_vec &
  operator-=(const sparse_vec<U> &b)
  {
    assert(this->tag_ == vec<T>::tag::STD);
    for (auto &p : b.data())
      this->std_repr_[p.first] -= p.second;
    return *this;
  }

  template <typename U>
  inline standard_vec &
  operator*=(U scale)
  {
    assert(this->tag_ == vec<T>::tag::STD);
    const size_t d = size();
    for (size_t i = 0; i < d; i++)
      this->std_repr_[i] *= scale;
    return *this;
  }

  inline standard_vec
  operator-()
  {
    standard_vec copy(*this);
    copy *= -1.0;
    return copy;
  }

  // this += (scale * v), but skip materializing the intermediate result
  template <typename U, typename V>
  inline standard_vec &
  add(U scale, const standard_vec<V> &v)
  {
    assert(this->tag_ == vec<T>::tag::STD);
    assert(size() == v.size());
    const size_t d = size();
    for (size_t i = 0; i < d; i++)
      this->std_repr_[i] += (scale * v[i]);
    return *this;
  }

  inline void
  clear()
  {
    assert(this->tag_ == vec<T>::tag::STD);
    this->std_repr_.clear();
  }

  inline void
  push_back(T t)
  {
    assert(this->tag_ == vec<T>::tag::STD);
    this->std_repr_.push_back(t);
  }

  // keeps the dimensions the same, but sets all values to 0
  inline void
  reset()
  {
    assert(this->tag_ == vec<T>::tag::STD);
    this->std_repr_.assign(size(), T());
  }

};

typedef standard_vec<double> standard_vec_t;

namespace ops {

// dot product

template <typename T1, typename T2>
static inline typename std::common_type<T1, T2>::type
dot(const standard_vec<T1> &a, const standard_vec<T2> &b)
{
  typedef typename std::common_type<T1, T2>::type T;
  assert(a.size() == a.size());
  const size_t d = a.size();
  T acc = T();
  for (size_t i = 0; i < d; i++)
    acc += a[i] * b[i];
  return acc;
}

template <typename T1, typename T2>
static inline typename std::common_type<T1, T2>::type
dot(const sparse_vec<T1> &a, const sparse_vec<T2> &b)
{
  // XXX: suboptimal implementation
  typedef typename std::common_type<T1, T2>::type T;
  T acc = T();
  const auto end = b.nonzero_elems().end();
  for (auto it = b.nonzero_elems().begin(); it != end; ++it)
    acc += a.get(it->first) * it->second;
  return acc;
}

template <typename T1, typename T2>
static inline typename std::common_type<T1, T2>::type
dot(const standard_vec<T1> &a, const sparse_vec<T2> &b)
{
  typedef typename std::common_type<T1, T2>::type T;
  T acc = T();
  const auto end = b.nonzero_elems().end();
  for (auto it = b.nonzero_elems().begin(); it != end; ++it)
    acc += a[it->first] * it->second;
  return acc;
}

template <typename T1, typename T2>
static inline typename std::common_type<T1, T2>::type
dot(const sparse_vec<T1> &a, const standard_vec<T2> &b)
{
  return dot(b, a);
}

template <typename T1, typename T2>
static inline typename std::common_type<T1, T2>::type
dot(const standard_vec<T1> &a, const vec<T2> &b)
{
  switch (b.get_tag()) {
  case vec<T2>::tag::STD:
    return dot(a, b.as_standard_ref());
  case vec<T2>::tag::SPARSE:
    return dot(a, b.as_sparse_ref());
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline typename std::common_type<T1, T2>::type
dot(const vec<T1> &a, const standard_vec<T2> &b)
{
  return dot(b, a);
}

template <typename T1, typename T2>
static inline typename std::common_type<T1, T2>::type
dot(const sparse_vec<T1> &a, const vec<T2> &b)
{
  switch (b.get_tag()) {
  case vec<T2>::tag::STD:
    return dot(a, b.as_standard_ref());
  case vec<T2>::tag::SPARSE:
    return dot(a, b.as_sparse_ref());
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline vec<typename std::common_type<T1, T2>::type>
dot(const vec<T1> &a, const sparse_vec<T2> &b)
{
  return dot(b, a);
}

template <typename T1, typename T2>
static inline typename std::common_type<T1, T2>::type
dot(const vec<T1> &a, const vec<T2> &b)
{
  switch (a.get_tag()) {
  case vec<T1>::tag::STD:
    return dot(a.as_standard_ref(), b);
  case vec<T1>::tag::SPARSE:
    return dot(a.as_sparse_ref(), b);
  }
  NOT_REACHABLE;
}

}

                    /** standard_vec operations */

template <typename T>
static inline std::ostream &
operator<<(std::ostream &o, const standard_vec<T> &v)
{
  o << v.data();
  return o;
}

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type>
operator+(const standard_vec<T1> &a, const standard_vec<T2> &b)
{
  typedef typename std::common_type<T1, T2>::type T;
  standard_vec<T> ret(a);
  return ret += b;
}

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type>
operator-(const standard_vec<T1> &a, const standard_vec<T2> &b)
{
  typedef typename std::common_type<T1, T2>::type T;
  standard_vec<T> ret(a);
  return ret -= b;
}

template <typename T>
T
standard_vec<T>::norm() const
{
  return sqrt(ops::dot(*this, *this));
}

template <typename T>
class sparse_vec : public vec<T> {
public:
  typedef std::pair<size_t, T> entry_type;
  typedef std::vector<entry_type> repr_type;

private:
  struct cmp {
    inline bool
    operator()(const entry_type &l, size_t r) const
    {
      return l.first < r;
    }
  };

  inline T &
  do_back_insert(size_t i)
  {
    this->sparse_repr_.emplace_back(i, T());
    return this->sparse_repr_.back().second;
  }

public:
  sparse_vec() : vec<T>(typename vec<T>::sparse_tag_t()) {}

  sparse_vec(const sparse_vec &) = default;
  sparse_vec &operator=(const sparse_vec &) = default;
  sparse_vec(sparse_vec &&) = default;

          /** vec api **/

  // insertion works best in ascending order
  inline T &
  ensureref(size_t i)
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    // fast-path
    if (this->sparse_repr_.empty() || this->sparse_repr_.back().first < i)
      return do_back_insert(i);

    // slow-path
    auto it = std::lower_bound(
        this->sparse_repr_.begin(),
        this->sparse_repr_.end(), i, cmp());
    assert(it != this->sparse_repr_.end());

    if (it->first == i)
      return it->second;

    // must insert
    return this->sparse_repr_.insert(it, entry_type(i, T()))->second;
  }

  inline T
  norm() const
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    T sum = T();
    for (auto &p : this->sparse_repr_)
      sum += p.second * p.second;
    return sqrt(sum);
  }

  inline T
  sum() const
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    T accum = T();
    for (auto p : this->sparse_repr_)
      accum += p.second;
    return accum;
  }

  inline void
  reserve(size_t n)
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    this->sparse_repr_.reserve(n);
  }

  inline size_t
  highest_nonzero_dim() const
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    if (this->sparse_repr_.empty())
      return 0;
    return this->sparse_repr_.back().first + 1;
  }

  inline size_t
  nnz() const
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    return this->sparse_repr_.size();
  }

          /** specific api **/

  inline T
  get(size_t i) const
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    if (this->sparse_repr_.empty() || this->sparse_repr_.back().first < i)
      return T();
    auto it = std::lower_bound(
        this->sparse_repr_.begin(),
        this->sparse_repr_.end(), i, cmp());
    assert(it != this->sparse_repr_.end());
    return (it->first == i) ? it->second : T();
  }

  inline const repr_type &
  nonzero_elems() const
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    return this->sparse_repr_;
  }

  inline repr_type &
  data()
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    return this->sparse_repr_;
  }

  inline const repr_type &
  data() const
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    return this->sparse_repr_;
  }

  template <typename U>
  inline sparse_vec &
  operator+=(const sparse_vec<U> &b)
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    for (auto &p : b.data())
      ensureref(p.first) += p.second;
    return *this;
  }

  template <typename U>
  inline sparse_vec &
  operator-=(const sparse_vec<U> &b)
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    for (auto &p : b.data())
      ensureref(p.first) -= p.second;
    return *this;
  }

  inline sparse_vec
  operator-()
  {
    sparse_vec v(*this);
    v *= -1.0;
    return v;
  }

  template <typename U>
  inline sparse_vec &
  operator*=(U scale)
  {
    assert(this->tag_ == vec<T>::tag::SPARSE);
    for (auto &p : data())
      p.second *= scale;
    return *this;
  }

};

typedef sparse_vec<double> sparse_vec_t;

                    /** sparse_vec operations */

template <typename T>
static inline std::ostream &
operator<<(std::ostream &o, const sparse_vec<T> &v)
{
  o << v.data();
  return o;
}

template <typename T1, typename T2>
static inline sparse_vec<typename std::common_type<T1, T2>::type>
operator+(const sparse_vec<T1> &a, const sparse_vec<T2> &b)
{
  typedef typename std::common_type<T1, T2>::type T;
  sparse_vec<T> ret(a);
  return ret += b;
}

template <typename T1, typename T2>
static inline sparse_vec<typename std::common_type<T1, T2>::type>
operator-(const sparse_vec<T1> &a, const sparse_vec<T2> &b)
{
  typedef typename std::common_type<T1, T2>::type T;
  sparse_vec<T> ret(a);
  return ret -= b;
}

                    /** mixed operations */

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type>
operator+(const standard_vec<T1> &a, const sparse_vec<T2> &b)
{
  typedef typename std::common_type<T1, T2>::type T;
  standard_vec<T> ret(a);
  return a += b;
}

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type>
operator+(const sparse_vec<T1> &a, const standard_vec<T2> &b)
{
  return b + a;
}

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type>
operator-(const standard_vec<T1> &a, const sparse_vec<T2> &b)
{
  typedef typename std::common_type<T1, T2>::type T;
  standard_vec<T> ret(a);
  return ret -= b;
}

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type>
operator-(const sparse_vec<T1> &a, const standard_vec<T2> &b)
{
  typedef typename std::common_type<T1, T2>::type T;
  standard_vec<T> ret(b);
  ret *= -1.0;
  return ret += a;
}

                    /** generic operations */

template <typename T>
inline standard_vec<T> *
vec<T>::as_standard_ptr()
{
  assert(tag_ == vec<T>::tag::STD);
  return static_cast<standard_vec<T> *>(this);
}

template <typename T>
inline const standard_vec<T> *
vec<T>::as_standard_ptr() const
{
  assert(tag_ == vec<T>::tag::STD);
  return static_cast<const standard_vec<T> *>(this);
}

template <typename T>
inline sparse_vec<T> *
vec<T>::as_sparse_ptr()
{
  assert(tag_ == vec<T>::tag::SPARSE);
  return static_cast<sparse_vec<T> *>(this);
}

template <typename T>
inline const sparse_vec<T> *
vec<T>::as_sparse_ptr() const
{
  assert(tag_ == vec<T>::tag::SPARSE);
  return static_cast<const sparse_vec<T> *>(this);
}

template <typename T>
inline standard_vec<T> &
vec<T>::as_standard_ref()
{
  return *as_standard_ptr();
}

template <typename T>
inline const standard_vec<T> &
vec<T>::as_standard_ref() const
{
  return *as_standard_ptr();
}

template <typename T>
inline sparse_vec<T> &
vec<T>::as_sparse_ref()
{
  return *as_sparse_ptr();
}

template <typename T>
inline const sparse_vec<T> &
vec<T>::as_sparse_ref() const
{
  return *as_sparse_ptr();
}

template <typename T>
inline T &
vec<T>::ensureref(size_t i)
{
  return tag_ == vec<T>::tag::STD ?
    as_standard_ref().ensureref(i) : as_sparse_ref().ensureref(i);
}

template <typename T>
inline T
vec<T>::norm() const
{
  return tag_ == vec<T>::tag::STD ?
    as_standard_ref().norm() : as_sparse_ref().norm();
}

template <typename T>
inline void
vec<T>::reserve(size_t n)
{
  tag_ == vec<T>::tag::STD ?
    as_standard_ref().reserve(n) : as_sparse_ref().reserve(n);
}

template <typename T>
inline size_t
vec<T>::highest_nonzero_dim() const
{
  return tag_ == vec<T>::tag::STD ?
    as_standard_ref().highest_nonzero_dim() : as_sparse_ref().highest_nonzero_dim();
}

template <typename T>
inline size_t
vec<T>::nnz() const
{
  return tag_ == vec<T>::tag::STD ?
    as_standard_ref().nnz() : as_sparse_ref().nnz();
}

template <typename T1, typename T2>
static inline vec<typename std::common_type<T1, T2>::type> &
operator*=(vec<T1> &a, T2 b)
{
  switch (a.get_tag()) {
  case vec<T1>::tag::STD:
    return a.as_standard_ref() *= b;
  case vec<T1>::tag::SPARSE:
    return a.as_sparse_ref() *= b;
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type>
operator+(const standard_vec<T1> &a, const vec<T2> &b)
{
  switch (b.get_tag()) {
  case vec<T2>::tag::STD:
    return a + b.as_standard_ref();
  case vec<T2>::tag::SPARSE:
    return a + b.as_sparse_ref();
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type>
operator+(const vec<T1> &a, const standard_vec<T2> &b)
{
  return b + a;
}

template <typename T1, typename T2>
static inline vec<typename std::common_type<T1, T2>::type>
operator+(const sparse_vec<T1> &a, const vec<T2> &b)
{
  switch (b.get_tag()) {
  case vec<T2>::tag::STD:
    return a + b.as_standard_ref();
  case vec<T2>::tag::SPARSE:
    return a + b.as_sparse_ref();
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline vec<typename std::common_type<T1, T2>::type>
operator+(const vec<T1> &a, const sparse_vec<T2> &b)
{
  return b + a;
}

template <typename T1, typename T2>
static inline vec<typename std::common_type<T1, T2>::type>
operator+(const vec<T1> &a, const vec<T2> &b)
{
  switch (a.get_tag()) {
  case vec<T1>::tag::STD:
    return a.as_standard_ref() + b;
  case vec<T1>::tag::SPARSE:
    return a.as_sparse_ref() + b;
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type> &
operator+=(standard_vec<T1> &a, const vec<T2> &b)
{
  switch (b.get_tag()) {
  case vec<T2>::tag::STD:
    return a += b.as_standard_ref();
  case vec<T2>::tag::SPARSE:
    return a += b.as_sparse_ref();
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type>
operator-(const standard_vec<T1> &a, const vec<T2> &b)
{
  switch (b.get_tag()) {
  case vec<T2>::tag::STD:
    return a - b.as_standard_ref();
  case vec<T2>::tag::SPARSE:
    return a - b.as_sparse_ref();
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type>
operator-(const vec<T1> &a, const standard_vec<T2> &b)
{
  switch (a.get_tag()) {
  case vec<T1>::tag::STD:
    return a.as_standard_ref() - b;
  case vec<T1>::tag::SPARSE:
    return a.as_sparse_ref() - b;
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline vec<typename std::common_type<T1, T2>::type>
operator-(const sparse_vec<T1> &a, const vec<T2> &b)
{
  switch (b.get_tag()) {
  case vec<T2>::tag::STD:
    return a - b.as_standard_ref();
  case vec<T2>::tag::SPARSE:
    return a - b.as_sparse_ref();
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline vec<typename std::common_type<T1, T2>::type>
operator-(const vec<T1> &a, const sparse_vec<T2> &b)
{
  switch (a.get_tag()) {
  case vec<T1>::tag::STD:
    return a.as_standard_ref() - b;
  case vec<T1>::tag::SPARSE:
    return a.as_sparse_ref() - b;
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline vec<typename std::common_type<T1, T2>::type>
operator-(const vec<T1> &a, const vec<T2> &b)
{
  switch (a.get_tag()) {
  case vec<T1>::tag::STD:
    return a.as_standard_ref() - b;
  case vec<T1>::tag::SPARSE:
    return a.as_sparse_ref() - b;
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline vec<typename std::common_type<T1, T2>::type>
operator*(const vec<T1> &a, T2 b)
{
  switch (a.get_tag()) {
  case vec<T1>::tag::STD:
    return a.as_standard_ref() * b;
  case vec<T1>::tag::SPARSE:
    return a.as_sparse_ref() * b;
  }
  NOT_REACHABLE;
}

template <typename T1, typename T2>
static inline vec<typename std::common_type<T1, T2>::type>
operator*(T1 a, const vec<T2> &b)
{
  return b * a;
}

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type>
operator*(const standard_vec<T1> &a, T2 b)
{
  typedef typename std::common_type<T1, T2>::type T;
  standard_vec<T> ret(a);
  return ret *= b;
}

template <typename T1, typename T2>
static inline standard_vec<typename std::common_type<T1, T2>::type>
operator*(T1 a, const standard_vec<T2> &b)
{
  return b * a;
}

template <typename T1, typename T2>
static inline sparse_vec<typename std::common_type<T1, T2>::type>
operator*(const sparse_vec<T1> &a, T2 b)
{
  typedef typename std::common_type<T1, T2>::type T;
  sparse_vec<T> ret(a);
  return ret *= b;
}

template <typename T1, typename T2>
static inline sparse_vec<typename std::common_type<T1, T2>::type>
operator*(T1 a, const sparse_vec<T2> &b)
{
  return b * a;
}

template <typename T>
static inline std::ostream &
operator<<(std::ostream &o, const vec<T> &v)
{
  switch (v.get_tag()) {
  case vec<T>::tag::STD:
    return o << v.as_standard_ref();
  case vec<T>::tag::SPARSE:
    return o << v.as_sparse_ref();
  }
  NOT_REACHABLE;
}
