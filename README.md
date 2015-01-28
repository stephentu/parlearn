parlearn
========

A simple parallel SGD implementation in C++11. 

Usage
-----
Here we assume that the code has been successfully compiled. Now let's run RCV-1. 
First, download the files from [here](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)

    $ mkdir -p data/rcv1
    $ cd data/rcv1
    
    # Just download testing data for now, since the training data is a bit larger. 
    $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2  
    $ bunzip2 rcv1_test.binary.bz2
    $ cd ../..
    $ src/cpp/converters/convert data/rcv1/rcv1_test.binary data/rcv1/rcv1_test.bin
    
    # Use actual training data when you are being serious
    $ src/cpp/tlearn --binary-training-file data/rcv1/rcv1_test.bin --binary-testing-file data/rcv1/rcv1_test.bin

And you will get something like
```
[INFO] PID=22333
[INFO] lambda=1e-05, rounds=1, offset=0, nworkers=1, lossfn=hinge, clf=CLF_SGD_NOLOCK
timed region `load training' took 2039.18 ms
[INFO] training set n=677399
timed region `load testing' took 1985.11 ms
[INFO] testing set n=677399
[INFO] training max norm 1
[INFO] fitting x_shape: {677399:47236}
[INFO] materializing took 0.003 ms
[INFO] max transformed norm is 1
[INFO] keep_histories: 0
[INFO] actual_nworkers: 1
[INFO] starting eta_t: 100000
[INFO] finished round 1 in 436.505 ms
[INFO] current risk: 60.7268
timed region `training phase' took 729.528 ms
evalution phase...
[INFO] w dim too large to print
[INFO] norm(w): 3476.17
[INFO] infnorm(w): 2503.4
[INFO] empirical risk: 60.7268
[INFO] norm gradient: 0.0348509
[INFO] classifier: {"clf_c0":"1.000000","clf_do_locking":"0","clf_name":"parsgd","clf_nrounds":"1","clf_nworkers":"1","clf_t_offset":"0","clf_training_sz":"677399","model_lambda":"0.000010","model_type":"linear"}
[INFO] acc on train: 0.969674
[INFO] acc on test: 0.969674
```
