#include <iostream>
#include <fstream>
#include <mkl.h>
#include "mpi.h"
#include <cassert>
#include <stdlib.h>
#include <cstdlib>
#include <errno.h>
#include <cstring>
#include <string.h>
#include <math.h>
#include <iomanip>
#include <vector>
#include <cmath>
#include <sys/time.h>

using namespace std;


int cols;
double val, x;
char* line;
int max_line_len = 1024*10;

// helper function to read dataset
static char* readline(FILE *input)
{
	int len;
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

// helper function to read LIBSVM format
void read(char* fname, int *rowidx,int *colidx,
	double *vals,double *y , int *nnz_local, int *m_local, int* nnzArray)
{

	int i;
	FILE * file;
	file = fopen(fname, "r");
	printf("%s\n",fname );


	if(file == NULL){
		printf("File not found!\n");
		return;
	}

	line= (char*) malloc(max_line_len*sizeof(char));

	int count_rowidx=0,count_colidx=0;;
	rowidx[count_rowidx]=1;
	i = 0;


	while (1)
	{
		if(readline(file)==NULL){
			break;
		}

		char *label, *value, *index, *endptr;
		label = strtok(line," \t\n");
		x = strtod(label, &endptr);
		while(1)
		{
			index = strtok(NULL,":");

			value = strtok(NULL," \t");

			if(value == NULL){
				break;
			}

			cols = (int) strtol(index,&endptr,10);
			colidx[count_colidx]=cols;
			nnzArray[cols-1]++;

			val =  strtod(value, &endptr);
			vals[count_colidx]=val;
			count_colidx++;
			i++;
		}
		count_rowidx++;
		rowidx[count_rowidx]=i+1;
		y[count_rowidx-1]=x;

	}
	*nnz_local = count_colidx;
	*m_local = count_rowidx;
	fclose(file);
}


// definition of the least sqaures objective function
// (||Ax-y||^2)/m+lambda*||x||
double objective_fun(double *x,
	double *vals,
	int *colidx,
	int *rowidx,
	double *y , int d, int m, double lambda){

	char *descr;
	descr=(char *)malloc(6*sizeof(char));

    // Descriptor of main sparse matrix properties
	descr[0]='G';
	descr[3]='F';
	double alpha = 1.0;
	double nalpha = -1.0;
	double zero =0.0;
	int inc = 1;
	int i;


	double *Ax = (double*) malloc (m*sizeof(double));

    // op: Ax = A*x
	char transa ='N';
	mkl_dcsrmv (&transa 
		, &m
		, &d
		, &alpha
		, descr
		, vals
		, colidx
		, rowidx
		, rowidx+1
		, x
		, &zero
		, Ax );

    // op:=> Ax - y
	cblas_daxpy (m, -1, y, 1, Ax, 1);

	// computing objective value
	double sum = 0;
	for (i = 0; i < d; ++i)
	{
		sum+= std::abs(x[i]);
	}
	sum  = sum + ddot(&m, Ax, &inc, Ax, &inc)/m + lambda * sum;

	free(Ax);

	return sum;
}

// main function for CASPNM
void CASPNM(	            int *rowidx,
	int *colidx,
	double *vals,
	int m,
	int n,
	double *y,
	int mlocal,
	double lambda,
	int s,
	double percent,
	int maxit,
	int seed,
	int freq,
	double *w,
	MPI_Comm comm,
	int Q,
	double beta,
	double gama_0)
{
	int npes, rank;
	MPI_Comm_size(comm, &npes);
	MPI_Comm_rank(comm, &rank);
	int b=std::floor(percent*mlocal);
	int len=b;
	double *alpha, *res,  *obj_err, *sol_err;
	double *del_w;
	double *wpre,*vec_tmp,*vec;
	double ctol;
	double *G, *recvG, *Xsamp, *wsamp, *sampres, *sampres_sum;
	int incx = 1;
	int rescnt = 1;
	int *index;
	int gram_size = n;
	int ngram = s*n*n;
	double wnorm=0;

	alpha = (double*) malloc (len*sizeof(double));
	G     = (double*) malloc (s*n*(n+1)*sizeof(double));
	alpha = (double*) malloc (len*sizeof(double));
	G     = (double*) malloc (s*n*(n+1)*sizeof(double));
	recvG = (double*) malloc (s*n*(n+1)*sizeof(double));
	del_w = (double*) malloc (s*n*sizeof(double));
	wsamp = (double*) malloc (n*sizeof(double));
	wpre  = (double*) malloc (n*sizeof(double));
	vec  = (double*) malloc (n*sizeof(double));
	vec_tmp  = (double*) malloc (n*sizeof(double));
	index = (int*) malloc (b*sizeof(int));
	sampres = (double*) malloc (n*sizeof(double));
	sampres_sum = (double*) malloc (n*sizeof(double));



	// storage for optimization variable
	int cumX_size = (maxit)/freq +20;
	struct timeval start,end;
	double *cumX;;
	long *times;

	

	for(int i = 0; i < n; ++i){
		sampres[i] = 1.;
		sampres_sum[i] = 1.;
	}
	
	memset(alpha, 0, sizeof(double)*len);
	memset(w, 0, sizeof(double)*n);
	memset(wpre, 0, sizeof(double)*n);
	memset(vec_tmp, 0, sizeof(double)*n);

	char transa = 'N', transb = 'T', uplo = 'U';
	double alp = 1.0/std::floor(percent*m);

	double one = 1., zero = 0., neg = -1.;
	int one_i=1;
	double neg_lambda = -lambda;
	double neg_alp = -alp;
	double resnrm = 1.;
	int info, nrhs = 1;
	double gama = gama_0;
	double tkp=1.0, tkn=1.0;
	int lGcols = b;


	char matdesc[6];
	matdesc[0] = 'G'; matdesc[3] = 'F';
	srand48(seed);

	double commst, commstp, commagg = 0.;
	double gramst, gramstp, gramagg = 0.;
	double innerst, innerstp, inneragg = 0.;
	int iter = 0;
	int offset = 0;

	int convcnt = 0;
	int conviter = 0;
	int convthresh = (n%b == 0) ? n/b : n/b + 1;

	int cursamp, count;
	vector<int> sampcolidx;
	vector<double> sampvals;
	vector<double> sampy;
	int cidx = 0, rnnz = 0;
	double tval = 0.;
	
	double grammax, innermax, commmax;
	gramst = MPI_Wtime();


	// initialize the storage for variables and time
	cumX = (double *) malloc((cumX_size*n)*sizeof(double));
	times = (long*) malloc((cumX_size)*sizeof(long));
	memcpy (cumX, w, n*sizeof(double)); 
	gettimeofday(&start,NULL);
	times[0] = 0;
	

// Generating b random numbers indicating samples that we want to choose
	while(1){
		for(int i = 0; i < s; ++i){
			vector<int> samprowidx(b+1, 1);

			cursamp = 0;
			count = 0;
			while(cursamp < b){
				if(((mlocal-count)*drand48()) >= (b - cursamp))
					++count;
				else{
					index[cursamp] = count;
					++count; ++cursamp;
				}
			}


	// Choosing b rows of the data matrix
			for(int k = 0; k < b; ++k){
				samprowidx[k+1]=samprowidx[k]+rowidx[index[k]+1]-rowidx[index[k]];
				sampy.push_back(y[index[k]]);
				for(int j =rowidx[index[k]]-1; j < rowidx[index[k]+1]-1; ++j){
					cidx = colidx[j];
					tval = vals[j];
					sampcolidx.push_back(cidx);
					sampvals.push_back(tval);
				}

			}
			// Computing G matrix which is: Y'Y where Y is the sampled data	
			mkl_dcsrmultd(&transb, &len, &gram_size, &gram_size, &sampvals[0], &sampcolidx[0], &samprowidx[0],  &sampvals[0], &sampcolidx[0], &samprowidx[0], G+i*n*n, &gram_size);
			int ngramtemp=n*n;

			// Multiply G by 1/m where m is #of samples
			dscal(&ngramtemp, &alp, G+i*n*n, &incx);

			// Computing alpha*A^T*x+beta*y where 
			// A is b*(s*n) matrix of sampled data[stored in samplval, sampcolidx and samprowidx]
			// beta is zero here
			// Therefore in following  line it's computing: X*y which is part of the gradient{G and R in our formulation}
			mkl_dcsrmv(&transb, &len, &gram_size, &alp, matdesc, &sampvals[0], &sampcolidx[0], &samprowidx[0], &samprowidx[1], &sampy[0], &zero, G+(s*n*n)+i*n);


			sampvals.clear();
			sampcolidx.clear();
			samprowidx.clear();
			sampy.clear();
		}


		gramstp = MPI_Wtime();
		gramagg += gramstp - gramst;

		// Reduce and Broadcast: Sum partial Gram and partial residual components.
		commst = MPI_Wtime();
		MPI_Allreduce(G,recvG,s*n*(n+1), MPI_DOUBLE, MPI_SUM, comm);
		commstp = MPI_Wtime();
		commagg += commstp - commst;
		innerst = MPI_Wtime();

		/*
		 * Inner s-step loop (unroll by 1 iteration so non-CA does not have a loop overhead).
		 * Perfomed redundantly on all processors
		*/
		// X'X*w-X*y or gradient is stored in recvG + s*n*(n+1)
		for(int in=0;in<Q;in++){
			memset(vec, 0, sizeof(double)*n);

			// CASPNM paramaters
			tkn=(1+sqrt(1+4*tkp*tkp))/2.0;
			double al1 = (tkp-1.0)/tkn;
			double al2 = -(tkp-1.0)/tkn;
			daxpy(&n, &al1 , w, &incx, vec, &incx);
			daxpy(&n, &one , w, &incx, vec, &incx);  
			daxpy(&n, &al2, wpre, &incx, vec, &incx); 

			for (int z = 0; z < n; ++z)
			{
				wpre[z]=w[z];
				w[z]=vec[z];
			}

			tkp=tkn;
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				n, 1, n, 1, recvG, n, w, 1, 0, vec_tmp, 1);
			daxpy(&n, &neg, recvG + s*n*n, &incx, vec_tmp, &incx);

			double neg_gama=-gama;
			daxpy(&n, &neg_gama, vec_tmp, &incx, w, &incx);

			// soft thresholding(applying norm-1 constraint):
			double thresh=lambda*gama;
			for (int i = 0; i < n; ++i)
			{
				if(w[i]>thresh)
					w[i]=w[i]-thresh;
				else if(w[i]<-thresh)
					w[i]=w[i]+thresh;
				else
					w[i]=0;
			}

			gama=gama*beta;
		}
		
		// store the optimization variable
		iter++;
		if(iter%freq==0){
			gettimeofday(&end,NULL);
			long seconds = end.tv_sec - start.tv_sec;
			times[iter/freq] = (seconds*1000)+(end.tv_usec - start.tv_usec)/1000;
			memcpy (cumX+(iter/freq )*n, w, n*sizeof(double)); 
		}

		innerstp = MPI_Wtime();
		inneragg += innerstp - innerst;	

		// if last iteration:
		if(iter == maxit ){
			/*Free stuff and collect timing stats from all ranks. (Lets be conservative and take running time of slowest processor)*/

			free(alpha); free(G); free(recvG);
			free(index); free(del_w); free(wsamp); 
			free(sampres);

			sampcolidx.clear(); sampvals.clear(); 
			MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

			if(rank == 0){
				cout << "Outer loop computation time: " << grammax*1000 << " ms" << endl;
				cout << "Inner loop computation time: " << innermax*1000 << " ms" << endl;
				cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << endl;
			}

			 // compute objective function and print it
			for (int i = 0; i < (iter-1)/freq; ++i)
			{
				double obj_value = objective_fun(cumX+i*n, vals,colidx,rowidx , y , n, mlocal, lambda);
				double obj_value_total = 0;
				MPI_Reduce(&obj_value, &obj_value_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
				obj_value_total = obj_value_total/npes;
				if(rank==0){
					printf(" %ld , %.8f \n", times[i], obj_value_total);
				}
			}

			return;
		}


		/*Start s-step inner loop, if s > 1 (i.e. we want the communication-avoiding routine).*/
		for(int i = 1; i < s; ++i){
			lGcols = i*n;

			// Compute residual based on previous subproblem solution
			innerst = MPI_Wtime();
			for(int in=0;in<Q;in++){
				memset(vec, 0, sizeof(double)*n);
				tkn=(1+sqrt(1+4*tkp*tkp))/2.0;
				double al1 = (tkp-1.0)/tkn;
				double al2 = -(tkp-1.0)/tkn;
				daxpy(&n, &al1 , w, &incx, vec, &incx);
				daxpy(&n, &one , w, &incx, vec, &incx);  
				daxpy(&n, &al2, wpre, &incx, vec, &incx); 	
				for (int z = 0; z < n; ++z)
				{
					wpre[z]=w[z];
					w[z]=vec[z];
				}

				tkp=tkn;

				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
					n, 1, n, 1, recvG + i*n*n, n, w, 1, 0, vec_tmp, 1);
				daxpy(&n, &neg, recvG + s*n*n+i*n, &incx, vec_tmp, &incx);
				double neg_gama=-gama;
				daxpy(&n, &neg_gama, vec_tmp, &incx, w, &incx);

				// soft thresholding
				double thresh=lambda*gama;
				for (int z = 0; z < n; ++z)
				{
					if(w[z]>thresh)
						w[z]=w[z]-thresh;
					else if(w[z]<-thresh)
						w[z]=w[z]+thresh;
					else
						w[z]=0;
				}

				gama=gama*beta;

			}

			iter++;

			// storing optimization vriable
			if(iter%freq==0){
				gettimeofday(&end,NULL);
				long seconds = end.tv_sec - start.tv_sec;
				times[iter/freq] = (seconds*1000)+(end.tv_usec - start.tv_usec)/1000;
				memcpy (cumX+(iter/freq)*n, w, n*sizeof(double)); 
			}

			inneragg += MPI_Wtime() - innerst;
			if(iter == maxit){
				free(alpha); free(G); free(recvG);
				free(index); free(del_w); free(wsamp); free(sampres);
				sampcolidx.clear(); sampvals.clear(); //sampres.clear();
				MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
				MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
				MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
				
				if(rank == 0){
					cout << "Outer loop computation time: " << grammax*1000 << " ms" << endl;
					cout << "Inner loop computation time: " << innermax*1000 << " ms" << endl;
					cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << endl;
				}

				// print objective function value
				for (int i = 0; i < (iter-1)/freq; ++i)
				{
					double obj_value = objective_fun(cumX+i*n, vals,colidx,rowidx , y , n, mlocal, lambda);
					double obj_value_total = 0;
					MPI_Reduce(&obj_value, &obj_value_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
					obj_value_total /=npes;
					if(rank==0){
						printf(" %ld , %.8f \n", times[i], obj_value_total);
					}			 
				}
				return;
			}

		}

		gramst = MPI_Wtime();
		gramagg += MPI_Wtime() - gramst;

		/* Reset some buffers and start next outer iteration */
		sampvals.clear();
		memset(G, 0, sizeof(double)*s*n*(n+1));
		memset(recvG, 0, sizeof(double)*s*n*(n+1));
		memset(del_w, 0, sizeof(double)*gram_size);
	}

}


int main (int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	int npes, rank;
	int m, n;
	char *pathName;
	double lambda,b;
	int maxit, seed, freq, s;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm comm = MPI_COMM_WORLD;

	// Print off a hello world message
	printf("rank %d: started\n", rank);

	if(npes<2){
		printf("%s\n", "There should be at least two processors!");
		MPI_Finalize();
		return 0;
	}

	if(argc < 14)
	{
		if(rank == 0)
		{
			std::cout << "Bad arg list!" << std::endl;
			std::cout << argv[0] << " [filepath] [nrows] [ncols] [lambda] [iterations] [freq] [sampling rate] [k] [number of benchmark iterations] [nnz] [initial step size] [Q] [beta]" << std::endl;
		}

		MPI_Finalize();
		return -1;
	}

	pathName = argv[1];
	m = atoi(argv[2]);
	n = atoi(argv[3]);
	lambda = atof(argv[4]);
	maxit = atoi(argv[5]);
	freq = atoi(argv[6]);
	b = atof(argv[7]);
	s = atoi(argv[8]);
	int niter = atoi(argv[9]);
	int nnz=atoi(argv[10]);
	double gama_0 = atof(argv[11]);
	int Q = atoi(argv[12]);
	double beta = atof(argv[13]);

	if(rank==0){

		cout<<"filepath: "<<argv[1]<<endl;
		cout<<"m: "<<argv[2]<<endl;
		cout<<"n: "<<argv[3]<<endl;
		cout<<"lambda: "<<argv[4]<<endl;
		cout<<"iterations: "<<argv[5]<<endl;
		cout<<"freq: "<<argv[6]<<endl;
		cout<<"sampling rate: "<<argv[7]<<endl;
		cout<<"s: "<<argv[8]<<endl;
		cout<<"#bechmark iterations: "<<argv[9]<<endl;
		cout<<"nnz: "<<argv[10]<<endl;
		cout<<"initial step size: "<<argv[11]<<endl;
		cout<<"Q: "<<argv[12]<<endl;
		cout<<"beta: "<<argv[13]<<endl;


	}

    int *rowidx, *colidx, *nnzArray, nnz_local, m_local; // m_local: number of samples for each processor
    double *vals,*y;

    // allocation memory for the sparse dataset
    rowidx=(int *)malloc((m+1)*sizeof(int)); 
    colidx=(int *)malloc(nnz*sizeof(int)); 
    vals=(double *)malloc(nnz*sizeof(double)); 
    y=(double *)malloc(m*sizeof(double)); 

    nnzArray=(int *)malloc(n*sizeof(int));
    memset(nnzArray,0,n*sizeof(int));

    char buf[100];
    strcat(pathName, "-");
    sprintf(buf, "%d", rank);
    strcat(pathName,buf);

	// read input file here
	// input dataset is always in LIBSVM format
    read(pathName, rowidx, colidx, vals, y, &nnz_local, &m_local, nnzArray);
    
    int dual_method = 0;

    if(rank == 0){
    	std::cout << "Finished reading the input dataset." << std::endl;
    }	

    double algst, algstp;
    double *w;
    w = (double*) malloc (n*sizeof(double));

    // A hard barrier before jumping into main loops
    MPI_Barrier(MPI_COMM_WORLD);

    int Kmax=3;
    int Jmax=3;
    double *results;
    results=(double *)malloc(Kmax*Jmax*sizeof(double));
    for(int k = 0; k < Kmax; ++k){
    	if(b > 1)
    		continue;
    	for(int j = 0; j < Jmax; ++j){


    		if(rank == 0){
    			std::cout << std::endl << std::endl;
    			std::cout << "s = " << s << ", " << "b = " << b << std::endl;
    		}
    		algst = MPI_Wtime();
    		for(int i = 0; i < niter; ++i){
    			seed = 0; 
    			CASPNM(rowidx, colidx, vals, m, n, y,m_local,lambda, s, b, maxit,
    				 seed, freq, w, comm, Q, beta, gama_0);
				seed++; //uncomment this for random runs everytime
    		}
    		algstp = MPI_Wtime();
    		double algmaxt = 1000*(algstp - algst)/niter;
    		double algmax=0;

    		MPI_Reduce(&algmaxt, &algmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    		if(rank == 0){
    			cout << endl << "Total CA-BCD time: " << algmax<< " ms" << endl;
    			results[k*Jmax+j]=algmax;

    		}
    		s *= 2;
    	}
    	s = 1;
    	b *= 2;
    }
    if(rank==0){
    	for (int k = 0; k < Kmax; ++k)
    	{
    		for (int j = 0; j < Jmax; ++j)
    		{
    			cout<<results[k*Jmax+j]<<" ";
    		}
    		cout<<endl;
    	}
    }

    MPI_Finalize();
    return 0;
}
