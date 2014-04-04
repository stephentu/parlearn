#include <getopt.h>
#include <unistd.h>

#include <iostream>
#include <cassert>
#include <vector>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <functional>

#include <ascii_file.hh>
#include <binary_file.hh>
#include <svmlight_file.hh>
#include <dataset.hh>
#include <vec.hh>
#include <pretty_printers.hh>
#include <loss_functions.hh>
#include <metrics.hh>
#include <timer.hh>
#include <gd.hh>
#include <sgd.hh>
#include <util.hh>

using namespace std;

typedef loss_functions::hinge_loss loss_fn;
typedef default_random_engine PRNG;

template <typename Clf>
static void
evalclf(const Clf &clf,
        const dataset &training,
        const dataset &testing)
{
  const auto train_predictions = clf.get_model().predict(training);
  const auto test_predictions  = clf.get_model().predict(testing);

  metrics::accuracy eval;
  const double train_acc = eval.score(training.get_y(), train_predictions);
  const double test_acc  = eval.score(testing.get_y(), test_predictions);

  if (clf.get_model().weightvec().size() <= 100)
    cout << "[INFO] w: " << clf.get_model().weightvec() << endl;
  else
    cout << "[INFO] w dim too large to print" << endl;
  cout << "[INFO] norm(w): " << clf.get_model().weightvec().norm() << endl;
  cout << "[INFO] infnorm(w): " << clf.get_model().weightvec().infnorm() << endl;
  cout << "[INFO] empirical risk: " << clf.get_model().empirical_risk(training) << endl;
  cout << "[INFO] norm gradient: " << clf.get_model().norm_grad_empirical_risk(training) << endl;
  cout << "[INFO] classifier: " << clf.jsonconfig() << endl;
  cout << "[INFO] acc on train: " << train_acc << endl;
  cout << "[INFO] acc on test: " << test_acc << endl;
}

template <typename Model, typename Loader>
static void
go(const string &training_file, const string &testing_file,
   function<Model(const dataset&, PRNG&)> builder,
   size_t nrounds, size_t nworkers,
   size_t offset, bool test_gd, Loader loader = Loader())
{
  typedef vector<vec_t> matrix_t;

  matrix_t xtrain, xtest;
  standard_vec_t ytrain, ytest;
  unsigned int nfeatures_train, nfeatures_test;
  {
    scoped_timer t("load training");
    if (loader.read_feature_file(training_file, xtrain, ytrain, nfeatures_train))
      throw runtime_error("could not read training file");
  }
  cout << "[INFO] training set n=" << xtrain.size() << endl;

  {
    scoped_timer t("load testing");
    if (loader.read_feature_file(testing_file, xtest, ytest, nfeatures_test))
      throw runtime_error("could not read testing file");
  }
  cout << "[INFO] testing set n=" << xtest.size() << endl;

  dataset training(move(xtrain), move(ytrain));
  dataset testing(move(xtest), move(ytest));
  training.set_parallel_materialize(true);
  testing.set_parallel_materialize(true);
  cout << "[INFO] training max norm " << training.max_x_norm() << endl;

  const unsigned seed =
    chrono::system_clock::now().time_since_epoch().count();
  shared_ptr<PRNG> prng(new PRNG(seed));

  Model model = builder(training, *prng);

  opt::gd<Model, PRNG> clf_gd(
      model, nrounds, prng, offset, 1.0, true);
  {
    scoped_timer t("training");
    clf_gd.fit(training);
  }

  cerr << "evaluting gd" << endl;
  evalclf(clf_gd, training, testing);

  opt::parsgd<Model, PRNG> clf_nolocking(
      model, nrounds, prng, nworkers, false, offset, 1.0, true);
  {
    scoped_timer t("training");
    clf_nolocking.fit(training);
  }

  cerr << "evaluting no-locking" << endl;
  evalclf(clf_nolocking, training, testing);

  opt::parsgd<Model, PRNG> clf_locking(
      model, nrounds, prng, nworkers, true, offset, 1.0, true);
  {
    scoped_timer t("training");
    clf_locking.fit(training);
  }

  cerr << "evaluting locking" << endl;
  evalclf(clf_locking, training, testing);
}

int
main(int argc, char **argv)
{
  string binary_training_file, binary_testing_file,
         ascii_training_file, ascii_testing_file,
         svmlight_training_file, svmlight_testing_file;
  double lambda = 1e-5;
  size_t nrounds = 1;
  size_t offset = 0;
  size_t nworkers = 1;
  bool test_gd = true;
  while (1) {
    static struct option long_options[] =
    {
      {"binary-training-file"   , required_argument , 0 , 'r'} ,
      {"binary-testing-file"    , required_argument , 0 , 't'} ,
      {"ascii-training-file"    , required_argument , 0 , 'a'} ,
      {"ascii-testing-file"     , required_argument , 0 , 'b'} ,
      {"svmlight-training-file" , required_argument , 0 , 'c'} ,
      {"svmlight-testing-file"  , required_argument , 0 , 'd'} ,
      {"lambda"                 , required_argument , 0 , 'l'} ,
      {"rounds"                 , required_argument , 0 , 'n'} ,
      {"offset"                 , required_argument , 0 , 'o'} ,
      {"threads"                , required_argument , 0 , 'w'} ,
      {0, 0, 0, 0}
    };
    int option_index = 0;
    int c = getopt_long(argc, argv, "r:t:a:b:c:d:l:n:o:w:", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
    case 0:
      if (long_options[option_index].flag != 0)
        break;
      abort();
      break;

    case 'r':
      binary_training_file = optarg;
      break;

    case 't':
      binary_testing_file = optarg;
      break;

    case 'a':
      ascii_training_file = optarg;
      break;

    case 'b':
      ascii_testing_file = optarg;
      break;

    case 'c':
      svmlight_training_file = optarg;
      break;

    case 'd':
      svmlight_testing_file = optarg;
      break;

    case 'l':
      lambda = strtod(optarg, nullptr);
      break;

    case 'n':
      nrounds = strtoul(optarg, nullptr, 10);
      break;

    case 'o':
      offset = strtoull(optarg, nullptr, 10);
      break;

    case 'w':
      nworkers = strtoull(optarg, nullptr, 10);
      break;

    default:
      abort();
    }
  }

  if ((!ascii_training_file.empty() +
       !binary_training_file.empty() +
       !svmlight_training_file.empty()) != 1)
    throw runtime_error("need exactly one of --ascii-training-file, "
        "--binary-training-file, or --svmlight-training-file");

  if ((!ascii_testing_file.empty() +
       !binary_testing_file.empty() +
       !svmlight_testing_file.empty()) != 1)
    throw runtime_error("need exactly one of --ascii-testing-file, "
        "--binary-testing-file, or --svmlight-testing-file");

  if (ascii_training_file.empty() != ascii_testing_file.empty() ||
      binary_training_file.empty() != binary_testing_file.empty())
    throw runtime_error("limitation: input file types must match for training and testing");

  if (lambda <= 0.0)
    throw runtime_error("need lambda > 0");
  if (nrounds <= 0)
    throw runtime_error("need rounds > 0");
  if (nworkers <= 0)
    throw runtime_error("need nworkers > 0");

  cerr << "[INFO] PID=" << getpid() << endl;
  cerr << "[INFO] lambda=" << lambda
       << ", rounds=" << nrounds
       << ", offset=" << offset
       << ", nworkers=" << nworkers
       << endl;

  using namespace std::placeholders;
  if (!ascii_training_file.empty()) {
    auto reg_builder =
      [lambda](const dataset &, PRNG &)
      { return model::linear_model<loss_fn>(lambda); };
    go<model::linear_model<loss_fn>, ascii_file>(
        ascii_training_file, ascii_testing_file,
        reg_builder, nrounds, nworkers, offset, test_gd);
  } else if (!binary_training_file.empty()) {
    auto reg_builder =
      [lambda](const dataset &, PRNG &)
      { return model::linear_model<loss_fn>(lambda); };
    go<model::linear_model<loss_fn>, binary_file>(
        binary_training_file, binary_testing_file,
        reg_builder, nrounds, nworkers, offset, test_gd);
  } else {
    ALWAYS_ASSERT(!svmlight_training_file.empty());
    auto reg_builder =
      [lambda](const dataset &, PRNG &)
      { return model::linear_model<loss_fn>(lambda); };
    go<model::linear_model<loss_fn>, svmlight_file>(
        svmlight_training_file, svmlight_testing_file,
        reg_builder, nrounds, nworkers, offset, test_gd);
  }

  return 0;
}
