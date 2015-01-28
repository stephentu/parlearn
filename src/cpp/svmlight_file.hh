#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include <vec.hh>

struct svmlight_file {

// not efficient, and not flexible (doesn't fully support
// the svmlight format)
//
// returns 0 on success, -1 on failure
//
// also currently loads in *sparse* format
int
read_feature_file(
    const std::string &filename,
    std::vector<vec_t> &xs, standard_vec_t &ys, unsigned int &n) const
{
  std::ifstream ifs(filename);
  std::string line, ns, ns0;
  std::vector<std::string> toks0, toks1;
  n = 0;
  while (std::getline(ifs, line)) {
		std::istringstream l(line);

    // class
    double y;
    l >> y;
    ALWAYS_ASSERT(y == 0.0 || y == 1.0 || y == -1.0);

    // NOTE: this namespace crap is only found in the VW-style modified
    // svmlight files, so let's just ignore it for now.
    //
    //// namespace
    //if (ns.empty()) {
    //  l >> ns;
    //} else {
    //  l >> ns0;
    //  if (ns != ns0)
    //    return -1;
    //}

    // features
    vec_t xv((vec_t::sparse_tag_t()));
    while (l.good()) {
      int c = l.get();
      if (isspace(c))
        continue;
      unsigned int i;
      l.unget();
      l >> std::skipws >> i >> std::ws;
      if (l.get() != ':')
        return -1;
      ALWAYS_ASSERT(i >= 1); // 1-based
      double x;
      l >> std::skipws >> x;
      xv.ensureref(i-1) = x;
      n = std::max(n, i);
    }

    xs.push_back(std::move(xv));
    ys.push_back(y == 0.0 ? -1.0 : y);
  }

  return 0;
}

};
