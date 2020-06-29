This is the implementation of paper **Reducing Communication in Proximal Newton Methods for Sparse Least Squares Problems**, proceedings of the 47th International Conference on Parallel Processing ([ICPP 2018](https://dl.acm.org/doi/abs/10.1145/3225058.3225131)). 

This is a high performance implementation in C using [MPI](https://mpitutorial.com/tutorials/). 



## Requirements

- Intel MKL 11.1.2
- MVAPICH2/2.1
- mpicc 14.0.2


## Compilation

First we have to set the environment variable MKLROOT. This depends on the path that MKL is installed. For default installation on LINUX systems: 

```sh
$ export MKLROOT=/opt/intel/mkl/
```

then we can compile the code using the provided makefile:
```sh
$ make
```


## Tests

CASPNM accepts multiple parameters as input:

```sh
mpirun -np [number of processors] [filepath] [nrows] [ncols] [lambda] [maxit] [freq] [sampling rate] [k] [number of benchmark iterations] [nnz] [initial step size] [Q] [beta]
```

- filepath: full path to the dataset
- nrows: number of samples in the dataset
- ncols: number of columns in the dataset
- lambda: regularization parameter
- iterations: number if iterations to run
- freq: frequency of computing the objective function value. 
- sampling rate: for stochastic gradient descent
- nnz: number of non-zeros in the dataset
- k, Q , beta: method parameters

A simple bash file is provided that can run mnist on 2 processors. In order to run, mnist dataset has to be split in two and put in a directory called dataset next to the code. Therefore,  dataset folder should contain mnist, mnist-0 and mnist-1 which are respectively the main dataset, the first split and the second split. You can find in the `scripts` directoy the required bash files. In order run:


```sh
$ sh mnistSplit.sh
$ sh run_mnist.sh
```


> **_NOTE:_** you can find all the scripts for tests in the `scripts` directory.


## How to split a dataset?

We provide a simple jar file that can split a dataset into arbitrary pieces. For example, to split the mnist dataset into 2 parts, you should put mnist in the dataset folder. Then, simply run the following:

```sh
$ java -jar Split.jar /path/to/dataset mnist 60000 2
```

To be more precise, Split.jar accepts the following parameters:

```sh
$ java -jar Split.jar path filename nrows nparts 
```

where path indicates the directory that contains the main dataset, filename is the dataset name, nrows is the number of rows in the dataset and nparts is the number of parts.

> **_NOTE:_** you can find `mnistSplit.sh` in the scripts folder and run `sh mnistSplit.sh`.

## Output

The output of the code contains the timings reported in the paper followd by two columns, which the first column is the time in milliseconds and the second column is the objective function value. 


## Troubleshooting

If you get an error as  `mpirun was unable to find the specified executable file...`, most likely it means that you have not compile the code properly. Make sure you have compiled the code using “make” command without any error.


If you get an error as `File not found!`, this means that one of the needed files for the dataset is not present. Make sure you put the dataset in the proper destination and you split it before running the code. Please refer to **How to split a dataset** section. 

