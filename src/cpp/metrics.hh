#pragma once

#include <macros.hh>
#include <vec.hh>

namespace metrics {

class accuracy {
public:
  inline double
  score(const standard_vec_t &actual, const standard_vec_t &predict) const
  {
    ALWAYS_ASSERT(actual.size() == predict.size());
    size_t correct = 0;
    for (size_t i = 0; i < actual.size(); i++)
      if (actual[i] == predict[i])
        correct++;
    return double(correct) / double(actual.size());
  }
};

} // namespace metrics
