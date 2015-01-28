parlearn
========

A simple parallel SGD implementation in C++11. 

Usage
-----
Here we assume that the code has been successfully compiled. Now let's run RCV-1. 
First, download the files from [here](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html)

    # Prepare the data
    $ mkdir -p data/rcv1
    $ cd data/rcv1
    $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2  
    $ wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2  
    $ bunzip2 rcv1_train.binary.bz2
    $ bunzip2 rcv1_test.binary.bz2
    $ cd ../..
    $ src/cpp/converters/convert data/rcv1/rcv1_train.binary data/rcv1/rcv1_train.bin
    $ src/cpp/converters/convert data/rcv1/rcv1_test.binary data/rcv1/rcv1_test.bin
    
    # Run SGD
    $ src/cpp/tlearn --binary-training-file data/rcv1/rcv1_train.bin --binary-testing-file data/rcv1/rcv1_test.bin --rounds 10

And you will get something like
```
[INFO] PID=22454
[INFO] lambda=1e-05, rounds=10, offset=0, nworkers=1, lossfn=hinge, clf=CLF_SGD_NOLOCK
timed region `load training' took 107.653 ms
[INFO] training set n=20242
timed region `load testing' took 1993.76 ms
[INFO] testing set n=677399
[INFO] training max norm 1
[INFO] fitting x_shape: {20242:47236}
[INFO] materializing took 0.002 ms
[INFO] max transformed norm is 1
[INFO] keep_histories: 0
[INFO] actual_nworkers: 1
[INFO] starting eta_t: 100000
[INFO] finished round 1 in 12.593 ms
[INFO] current risk: 124785
[INFO] finished round 2 in 12.63 ms
[INFO] current risk: 27924.7
[INFO] finished round 3 in 12.67 ms
[INFO] current risk: 12419.4
[INFO] finished round 4 in 12.717 ms
[INFO] current risk: 6844.67
[INFO] finished round 5 in 12.619 ms
[INFO] current risk: 4359.93
[INFO] finished round 6 in 12.666 ms
[INFO] current risk: 2992.22
[INFO] finished round 7 in 12.776 ms
[INFO] current risk: 2175.91
[INFO] finished round 8 in 12.605 ms
[INFO] current risk: 1660.11
[INFO] finished round 9 in 12.663 ms
[INFO] current risk: 1305.96
[INFO] finished round 10 in 12.785 ms
[INFO] current risk: 1053.91
timed region `training phase' took 170.836 ms
evalution phase...
[INFO] w dim too large to print
[INFO] norm(w): 14513.4
[INFO] infnorm(w): 9748.31
[INFO] empirical risk: 1053.91
[INFO] norm gradient: 0.145194
[INFO] classifier: {"clf_c0":"1.000000","clf_do_locking":"0","clf_name":"parsgd","clf_nrounds":"10","clf_nworkers":"1","clf_t_offset":"0","clf_training_sz":"20242","model_lambda":"0.000010","model_type":"linear"}
[INFO] acc on train: 0.9917
[INFO] acc on test: 0.942836
```
