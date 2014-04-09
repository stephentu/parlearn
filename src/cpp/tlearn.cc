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
using namespace model;
using namespace loss_functions;

typedef default_random_engine PRNG;
typedef vector<vec_t> matrix_t;

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

template <typename Clf>
static void
execclf(Clf &clf, const dataset &training, const dataset &testing)
{
  {
    scoped_timer t("training phase");
    clf.fit(training);
  }
  cerr << "evalution phase..." << endl;
  evalclf(clf, training, testing);
}

enum class ClfType { CLF_GD, CLF_SGD_NOLOCK, CLF_SGD_LOCK };

// why isn't this auto-generated?
static const char *
clftype_str(ClfType t)
{
  switch (t) {
  case ClfType::CLF_GD: return "CLF_GD";
  case ClfType::CLF_SGD_NOLOCK: return "CLF_SGD_NOLOCK";
  case ClfType::CLF_SGD_LOCK: return "CLF_SGD_LOCK";
  default: return nullptr;
  }
}

template <typename LossFn>
static void
go(const dataset &training, const dataset &testing,
   ClfType clftype, double lambda,
   size_t nrounds, size_t nworkers, size_t offset)
{
  const unsigned seed =
    chrono::system_clock::now().time_since_epoch().count();
  shared_ptr<PRNG> prng(new PRNG(seed));

  typedef linear_model<LossFn> Model;
  Model model(lambda);

  if (clftype == ClfType::CLF_GD) {
    opt::gd<Model, PRNG> clf(
        model, nrounds, prng, offset, 1.0, true);
    execclf(clf, training, testing);
  } else if (clftype == ClfType::CLF_SGD_NOLOCK) {
    opt::parsgd<Model, PRNG> clf(
        model, nrounds, prng, nworkers, false, offset, 1.0, true);
    execclf(clf, training, testing);
  } else /* if (clftype == ClfType::CLF_SGD_LOCK) */ {
    opt::parsgd<Model, PRNG> clf(
        model, nrounds, prng, nworkers, true, offset, 1.0, true);
    execclf(clf, training, testing);
  }
}

template <typename Loader>
static void
load(const string &training_file, const string &testing_file,
     matrix_t &xtrain, standard_vec_t &ytrain,
     matrix_t &xtest, standard_vec_t &ytest,
     Loader loader = Loader())
{
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
}

int
main(int argc, char **argv)
{
  string binary_training_file, binary_testing_file,
         ascii_training_file, ascii_testing_file,
         svmlight_training_file, svmlight_testing_file;
  string lossfn = "hinge";
  ClfType clftype = ClfType::CLF_SGD_NOLOCK;
  double lambda = 1e-5;
  size_t nrounds = 1;
  size_t offset = 0;
  size_t nworkers = 1;
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
      {"loss"                   , required_argument , 0 , 'f'} ,
      {"clf"                    , required_argument , 0 , 'g'} ,
      {0, 0, 0, 0}
    };
    int option_index = 0;
    int c = getopt_long(argc, argv, "r:t:a:b:c:d:l:n:o:w:f:g:", long_options, &option_index);
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

    case 'f':
      lossfn = optarg;
      break;

    case 'g':
      {
        string o(optarg);
        if (o == "gd")
          clftype = ClfType::CLF_GD;
        else if (o == "sgd-nolock")
          clftype = ClfType::CLF_SGD_NOLOCK;
        else if (o == "sgd-lock")
          clftype = ClfType::CLF_SGD_LOCK;
        else
          throw runtime_error("Invalid clf: " + o);
      }
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

  if (lossfn != "logistic" && lossfn != "square" &&
      lossfn != "hinge" && lossfn != "ramp")
    throw runtime_error("invalid loss function: " + lossfn);

  cerr << "[INFO] PID=" << getpid() << endl;
  cerr << "[INFO] lambda=" << lambda
       << ", rounds=" << nrounds
       << ", offset=" << offset
       << ", nworkers=" << nworkers
       << ", lossfn=" << lossfn
       << ", clf=" << clftype_str(clftype)
       << endl;

  // load the dataset
  matrix_t xtrain, xtest;
  standard_vec_t ytrain, ytest;
  if (!ascii_training_file.empty())
    load<ascii_file>(ascii_training_file, ascii_testing_file,
                     xtrain, ytrain, xtest, ytest);
  else if (!binary_training_file.empty())
    load<binary_file>(binary_training_file, binary_testing_file,
                      xtrain, ytrain, xtest, ytest);
  else /* if (!svmlight_training_file.empty()) */
    load<svmlight_file>(svmlight_training_file, svmlight_testing_file,
                        xtrain, ytrain, xtest, ytest);

  dataset training(move(xtrain), move(ytrain));
  dataset testing(move(xtest), move(ytest));
  training.set_parallel_materialize(true);
  testing.set_parallel_materialize(true);
  cout << "[INFO] training max norm " << training.max_x_norm() << endl;

  // build the model
  if (lossfn == "logistic")
    go<logistic_loss>(training, testing, clftype, lambda, nrounds, nworkers, offset);
  else if (lossfn == "square")
    go<square_loss>(training, testing, clftype, lambda, nrounds, nworkers, offset);
  else if (lossfn == "hinge")
    go<hinge_loss>(training, testing, clftype, lambda, nrounds, nworkers, offset);
  else /* if (lossfn == "ramp") */
    go<ramp_loss>(training, testing, clftype, lambda, nrounds, nworkers, offset);

  return 0;
}
