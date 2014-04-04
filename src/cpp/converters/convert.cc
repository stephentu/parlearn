/**
 * convert.cc - converts an svmlight file to binary file
 */

#include <iostream>
#include <string>
#include <vector>

#include <binary_file.hh>
#include <svmlight_file.hh>
#include <vec.hh>

using namespace std;

int
main(int argc, char **argv)
{
  if (argc != 3) {
    cerr << "[usage] " << argv[0] << " svmlight_file binary_file" << endl;
    return 1;
  }

  vector<vec_t> xs;
  standard_vec_t ys;
  unsigned int n;
  if (svmlight_file().read_feature_file(argv[1], xs, ys, n)) {
    cerr << "[ERROR] could not read svmlight_file" << endl;
    return 1;
  }

  if (binary_file().write_feature_file(argv[2], xs, ys, true)) {
    cerr << "[ERROR] could not write binary_file" << endl;
    return 1;
  }

  return 0;
}
