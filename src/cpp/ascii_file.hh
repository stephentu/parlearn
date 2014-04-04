#pragma once

#include <cassert>
#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include <vec.hh>

struct ascii_file {

/**
 * Currently loads in dense vector format
 */
int
read_feature_file(
    const std::string &filename,
    std::vector<vec_t> &xs, standard_vec_t &ys, unsigned int &n) const
{
  std::ifstream ifs(filename);
  std::string line;
  n = 0;
  while (std::getline(ifs, line)) {
		std::istringstream l(line);

    // class
    double y;
    l >> y;
    assert(y == -1.0 || y == 1.0);
    ys.as_standard_ref().data().push_back(y);

    vec_t xv; // dense vec
    xv.reserve(n);
    while (l.good()) {
      double x;
      l >> x >> std::ws;
      xv.as_standard_ref().data().push_back(x);
    }
    n = std::max(size_t(n), xv.as_standard_ref().size());
    xs.push_back(std::move(xv));
  }
  assert(xs.size() == ys.size());
  return (ifs.peek() == EOF) ? 0 : -1;
}

};
