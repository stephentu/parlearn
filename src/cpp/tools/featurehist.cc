#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <dataset.hh>
#include <binary_file.hh>

using namespace std;

int
main(int argc, char **argv)
{
  if (argc != 3) {
    cerr << "[usage] " << argv[0] << " binary_file output_file" << endl;
    return 1;
  }

  vector<vec_t> xs;
  standard_vec_t ys;
  unsigned int n;
  if (binary_file().read_feature_file(argv[1], xs, ys, n)) {
    cerr << "[ERROR] could not read binary_file" << endl;
    return 1;
  }

  dataset d(move(xs), move(ys));
  d.set_parallel_materialize(true);
  const auto shape = d.get_x_shape();

  vector<size_t> counts(shape.second);
  const auto it_end = d.x_end();
  for (auto it = d.x_begin(); it != it_end; ++it) {
    const auto &x = *it;
    const auto inner_it_end = x.end();
    for (auto inner_it = x.begin(); inner_it != inner_it_end; ++inner_it)
      counts[inner_it.tell()]++;
  }

  ofstream ofs(argv[2]);
  for (auto c : counts)
    ofs << c << endl;

  return 0;
}

