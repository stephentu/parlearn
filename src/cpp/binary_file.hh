#pragma once

#include <cassert>
#include <cstdint>
#include <fstream>
#include <string>
#include <stdexcept>

#include <vec.hh>

struct binary_file_header {
  enum class type : uint8_t {
    BINARY_FILE_DENSE = 0x1,
    BINARY_FILE_SPARSE,
  };
  type t;
} __attribute__((packed)) ;

struct binary_file {

template <typename T>
static inline bool
read_from_istream(std::istream &is, T &t)
{
  is.read((char *) &t, sizeof(t));
  return is.good();
}

static inline bool
read_feature_vector(
    std::istream &is, vec_t &xv,
    uint32_t num_features, bool sparse_format)
{
  for (size_t i = 0; i < num_features; i++) {
    uint32_t feature_idx;
    double value;
    if (sparse_format && !read_from_istream(is, feature_idx))
      throw std::runtime_error("could not read feature idx");
    else if (!sparse_format)
      feature_idx = i;
    if (!read_from_istream(is, value))
      throw std::runtime_error("could not read value");
    xv.ensureref(feature_idx) = value;
  }
  return true;
}

static inline bool
is_sparse_feature_file(const std::string &filename)
{
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (!ifs.good())
    throw std::runtime_error("could not open file");
  binary_file_header hdr;
  if (!read_from_istream(ifs, hdr))
    throw std::runtime_error("bad header");
  return hdr.t == binary_file_header::type::BINARY_FILE_SPARSE;
}

// returns -1 on failure, 0 on success
int
read_feature_file(
    const std::string &filename,
    std::vector<vec_t> &xs, standard_vec_t &ys, unsigned int &n) const
{
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (!ifs.good())
    throw std::runtime_error("could not open file");

  // header
  binary_file_header hdr;
  if (!read_from_istream(ifs, hdr))
    throw std::runtime_error("bad header");

  // sparse_format: sparse_line*
  // sparse_line:
  //   [classification (int8_t) |
  //    num_features (uint32_t) |
  //    [feature_idx (uint32_t) | value (double)]* (num_features repetitions)]
  //
  // dense_format: num_features (uint32_t) dense_line*
  // dense_line:
  //   [classification (int8_t) |
  //    [value (double)]* (num_features repetitions)]

  if (hdr.t == binary_file_header::type::BINARY_FILE_SPARSE) {
    while (ifs.good()) {
      if (ifs.peek() == EOF)
        break;
      int8_t classification;
      uint32_t num_features;
      if (!read_from_istream(ifs, classification) ||
          !read_from_istream(ifs, num_features))
        throw std::runtime_error("bad sparse feature vector desc");
      vec_t xv((vec_t::sparse_tag_t()));
      xv.reserve(num_features);
      if (!read_feature_vector(ifs, xv, num_features, true))
        throw std::runtime_error("bad sparse feature vector");
      n = std::max(xv.highest_nonzero_dim(), static_cast<size_t>(n));
      xs.push_back(std::move(xv));
      ys.data().push_back(
          static_cast<double>(static_cast<int32_t>(classification)));
    }
  } else {
    uint32_t num_features;
    if (!read_from_istream(ifs, num_features))
      throw std::runtime_error("bad dense format");
    n = num_features;
    while (ifs.good()) {
      if (ifs.peek() == EOF)
        break;
      int8_t classification;
      vec_t xv;
      if (!read_from_istream(ifs, classification) ||
          !read_feature_vector(ifs, xv, num_features, false))
        throw std::runtime_error("bad dense feature vector");
      xs.push_back(std::move(xv));
      ys.data().push_back(
          static_cast<double>(static_cast<int32_t>(classification)));
    }
  }

  return (ifs.peek() == EOF) ? 0 : -1;
}

template <typename T>
static inline bool
write_to_ostream(std::ostream &o, const T &t)
{
  o.write((const char *) &t, sizeof(t));
  return o.good();
}

// returns -1 on failure, 0 on success
int
write_feature_file(
    const std::string &filename,
    const std::vector<vec_t> &xs,
    const standard_vec_t &ys,
    bool sparse_format) const
{
  std::ofstream ofs(filename);

  binary_file_header hdr;
  hdr.t = sparse_format ?
    binary_file_header::type::BINARY_FILE_SPARSE :
    binary_file_header::type::BINARY_FILE_DENSE;

  if (!write_to_ostream(ofs, hdr))
    return -1;

  if (sparse_format) {
    for (size_t i = 0; i < xs.size(); i++) {
      const auto &xv = xs[i];
      const int8_t classification =
        static_cast<int32_t>(ys[i]);
      const uint32_t num_features = xv.nnz();
      if (!write_to_ostream(ofs, classification) ||
          !write_to_ostream(ofs, num_features))
        return -1;
      const auto end = xv.end();
      for (auto it = xv.begin(); it != end; ++it) {
        const uint32_t feature_idx = it.tell();
        if (!write_to_ostream(ofs, feature_idx) ||
            !write_to_ostream(ofs, *it))
          return -1;
      }
    }
  } else {
    // XXX: cannot currently support writing sparse vectors in
    // dense format
    const uint32_t num_features = xs.empty() ? 0 : xs.front().nnz();
    if (!write_to_ostream(ofs, num_features))
      return -1;
    for (size_t i = 0; i < xs.size(); i++) {
      const auto &xv = xs[i];
      assert(xv.nnz() == num_features);
      const int8_t classification =
        static_cast<int32_t>(ys[i]);
      if (!write_to_ostream(ofs, classification))
        return -1;
      for (auto v : xv.as_standard_ref().data())
        if (!write_to_ostream(ofs, v))
          return -1;
    }
  }

  return ofs.good() ? 0 : -1;
};

};
