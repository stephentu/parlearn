#pragma once

#include <vec.hh>
#include <random>

namespace util {

template <typename T, typename Generator>
static inline void
inplace_symmetric_multivariate_normal(standard_vec<T> &v, Generator &prng, double sigma, size_t d)
{
  std::normal_distribution<T> gauss(0.0, 1.0);
  v.resize(d);
  for (size_t i = 0; i < d; i++)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    v[i] = sigma * gauss(prng);
}

/**
 * z ~ Normal(0, sigma^2 * eye(d))
 */
template <typename T, typename Generator>
static inline standard_vec<T>
symmetric_multivariate_normal(Generator &prng, double sigma, size_t d)
{
  standard_vec<T> v;
  inplace_symmetric_multivariate_normal(v, prng, sigma, d);
  return v;
}

template <typename T, typename Generator>
static inline size_t
sample_masses_cdf(Generator &prng, const standard_vec<T> &cdf)
{
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  const double u = unif(prng);
  for (size_t i = 0; i < cdf.size(); i++)
    if (u <= cdf[i])
      return i;
  ALWAYS_ASSERT(false);
  return 0;
}

template <typename VT>
static inline standard_vec<typename VT::value_type>
dimslice(const std::vector<VT> &vs,
         size_t dim, size_t first, size_t last)
{
  std::vector<typename VT::value_type> ret;
  ret.reserve(last - first);
  for (size_t i = first; i < last; i++)
    ret.push_back(vs[i][dim]);
  return standard_vec<typename VT::value_type>(std::move(ret));
}

template <typename VT>
static inline standard_vec<typename VT::value_type>
mean(const std::vector<VT> &vs, size_t first, size_t last)
{
  standard_vec<typename VT::value_type> ret(vs.front().size());
  for (size_t dim = 0; dim < ret.size(); dim++)
    ret[dim] = dimslice(vs, dim, first, last).mean();
  return ret;
}

template <typename VT>
static inline std::vector<standard_vec<typename VT::value_type>>
cumsum(const std::vector<VT> &vs, size_t first, size_t last)
{
  std::vector<standard_vec<typename VT::value_type>> ret(vs.front().size());
  for (size_t dim = 0; dim < ret.size(); dim++)
    ret[dim] = dimslice(vs, dim, first, last).cumsum();
  return ret;
}

} // namespace util
