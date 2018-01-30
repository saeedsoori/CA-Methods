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
//#include "svm.h"

using namespace std;
//#include "cabcd.h"
//#include "util.h"

int cols;
double val, x;
char* line;
int max_line_len;
ofstream myfile;
ifstream wopfile;

void write_to_file(double *w, int n){
	//myfile<<"1 ";
	for (int i = 0; i < n; ++i)
	{
		if(w[i]<0.00001 & w[i]>-0.00001)
			continue;
		else
			myfile<<i+1<<":"<<w[i]<< " ";
	}
	myfile<<endl;
}
void printme(double *w, int n){
	//myfile<<"1 ";
	for (int i = 0; i < n; ++i)
	{
		if(w[i]<0.00001 & w[i]>-0.00001)
			continue;
		else
			cout<<i+1<<":"<<w[i]<< " ";
	}
	cout<<endl;
}
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

void read(char* fname, int *rowidx,int *colidx,double *vals,double *y)
{
  int i, j;
  FILE * file;
  file = fopen(fname, "r");
  max_line_len = 1024*10;
  line = (char*) malloc (max_line_len*sizeof(char));
  int count_rowidx=0,count_colidx=0;;
  rowidx[count_rowidx]=1;
        i = 0;

  while (readline(file) != NULL)
    {
    		 // cout << "there" <<endl;

      char *label, *value, *index, *endptr;
      label = strtok(line," \t\n");
      x = strtod(label, &endptr);
      //cout << x << " ";
      while(1)
	{
	  index = strtok(NULL,":");
	  value = strtok(NULL," \t");
	  if(value == NULL)
	    break;
	  //cout << "here" <<endl;
	  cols = (int) strtol(index,&endptr,10);
	  //colidx.push_back(cols-1);
	  colidx[count_colidx]=cols;


	  val = strtod(value, &endptr);
	  //vals.push_back(val);
	  vals[count_colidx]=val;

	  	  count_colidx++;

	  //cout << colidx[j] << " " << vals[j] << endl;
	  i++;
	}
      count_rowidx++;
      rowidx[count_rowidx]=i+1;
      //cout << "here" << endl;
      //y.push_back(x);
      y[count_rowidx-1]=x;
    }
  fclose(file);
}



void Prox(double *w,int n,int lambda){
	for (int i = 0; i < n; ++i)
	{
		if(w[i]>lambda)
			w[i]=w[i]-lambda;
		else if(w[i]<-lambda)
			w[i]=w[i]+lambda;
		else
			w[i]=0;
	}

}

      
void cabcd(	            int *rowidx,
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
						double tol,
						int seed,
						int freq,
						double *w,
						MPI_Comm comm,
						double *wop,
						int Q,
						double beta,
						double gama_0)
{
	int npes, rank;
	MPI_Comm_size(comm, &npes);
	MPI_Comm_rank(comm, &rank);
//if(s>1)
  //                      printf("rank==%d , len==%d \n",rank,len);
//MPI_Barrier(comm);
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

	/*assert(0==Malloc_aligned(double, alpha, len, ALIGN));
	assert(0==Malloc_aligned(double, G, gram_size*(gram_size + 2), ALIGN));
	assert(0==Malloc_aligned(double, recvG, s*b*(s*b + 2), ALIGN));
	assert(0==Malloc_aligned(double, del_w, s*b, ALIGN));
	assert(0==Malloc_aligned(double, wsamp, s*b, ALIGN));
	assert(0==Malloc_aligned(int, index, s*b, ALIGN));
	assert(0==Malloc_aligned(double, sampres, n, ALIGN));
	assert(0==Malloc_aligned(double, sampres_sum, n, ALIGN));*/

	for(int i = 0; i < n; ++i){
		sampres[i] = 1.;
		sampres_sum[i] = 1.;
	}
	//printf("rank=: %d A\n",rank);
	//MPI_Barrier(comm);
	//printf("A\n");
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
	///////
	////
	int lGcols = b;
	///
	///////
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
	// m+1 should change in parallel mode
		//vector<int> samprowidx(s*b+1, 1);
	//vector<int> samprowidx(b+1, 1);
	vector<int> sampcolidx;
	vector<double> sampvals;
	vector<double> sampy;
	int cidx = 0, rnnz = 0;
	double tval = 0.;
	
	//std::vector<double> sampres(n, 1.);
	double grammax, innermax, commmax;
	gramst = MPI_Wtime();

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
					//if(rank == 0)
						//std::cout << count << ' ';
					++count; ++cursamp;
				}
			}
		
	
	// Choosing b rows of the data matrix
			//samprowidx[0]=1;
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
			//printme(G+i*n*n,n*n);	

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

		/*int rr=0;
		for(int i = 0; i < gram_size; ++i){
			wsamp[i] = w[rr];
	 		rr++;
	 		if(rr>n-1)
	 			rr=0;

	 	}*/
	
	 	gramstp = MPI_Wtime();
		gramagg += gramstp - gramst;
		// Reduce and Broadcast: Sum partial Gram and partial residual components.
	 	commst = MPI_Wtime();
	 	MPI_Allreduce(G,recvG,s*n*(n+1), MPI_DOUBLE, MPI_SUM, comm);
	 	commstp = MPI_Wtime();
	 	commagg += commstp - commst;
	 	innerst = MPI_Wtime();

		// We don't need the lambda here since it is L1 norm and non-differentiable
		//for(int i =0; i < s*n; ++i)
		//		recvG[i + i*s*n] += lambda;

		/*
		 * Inner s-step loop (unroll by 1 iteration so non-CA does not have a loop overhead).
		 * Perfomed redundantly on all processors
		*/
		// X'X*w-X*y or gradient is stored in recvG + s*n*(n+1)
	 	for(int in=0;in<Q;in++){
	 		/*cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                n, 1, n, 1, recvG, n, w, 1, 0, vec_tmp, 1);
	 		daxpy(&n, &neg, recvG + s*n*n, &incx, vec_tmp, &incx);*/
			memset(vec, 0, sizeof(double)*n);
			tkn=(1+sqrt(1+4*tkp*tkp))/2.0;
	 		double al1 = (tkp-1.0)/tkn;
			double al2 = -(tkp-1.0)/tkn;
		//	double al1=(iter-2)/(iter+1);
		 //	double al2=-(iter-2)/(iter+1);
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

			// do soft thresholding:
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
		


	if(iter%freq==0){
		for (int z = 0; z < n; ++z)
		{
			wsamp[z]=w[z];
		}
		daxpy(&n, &neg, wop, &incx, wsamp, &incx);
		resnrm=dnrm2(&n, wsamp, &incx)/dnrm2(&n, wop, &incx);
		if(resnrm <= tol){
					free(alpha); free(G); free(recvG);
					free(index); free(del_w); free(wsamp); free(sampres);
					sampcolidx.clear(); sampvals.clear();// sampres.clear();
					MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
					MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
					MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
					if(rank == 0){
						cout << "CA-BCD converged with residual: " << scientific << resnrm << setprecision(4) << fixed << " At iteration: " << iter << endl;
						cout << "Outer loop computation time: " << grammax*1000 << " ms" << endl;
						cout << "Inner loop computation time: " << innermax*1000 << " ms" << endl;
						cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << endl;
					}
					return;
			
			}
	}
	 	//printf("w after proximal at iter: %d\n",iter );
	 	//printme(w, n);


		iter++;
	 	innerstp = MPI_Wtime();
	 	inneragg += innerstp - innerst;	
		if(iter == maxit ){

		//	printf("rank=: %d E\n",rank);			
		//	MPI_Barrier(comm);
			/*Free stuff and collect timing stats from all ranks. (Lets be conservative and take running time of slowest processor)*/

			free(alpha); free(G); free(recvG);
			free(index); free(del_w); free(wsamp); 
			free(sampres);
			//cout<<"line 335"<<endl;
			//samprowidx.clear();
			 sampcolidx.clear(); sampvals.clear(); //sampres.clear();
		 	MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			//cout<<"line 338"<<endl;
		 	MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			//cout<<"line 340"<<endl;
		 	MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			//cout<<"line 342"<<endl;
		 	if(rank == 0){
				cout << "Outer loop computation time: " << grammax*1000 << " ms" << endl;
				cout << "Inner loop computation time: " << innermax*1000 << " ms" << endl;
		 		cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << endl;
		 	}
			//cout<<"line 348"<<endl;
//			myfile.close();
			return;
		}
		

		/*Start s-step inner loop, if s > 1 (i.e. we want the communication-avoiding routine).*/
		
		for(int i = 1; i < s; ++i){
			lGcols = i*n;
			// Compute residual based on previous subproblem solution
			innerst = MPI_Wtime();
			/////
			for(int in=0;in<Q;in++){
	 /*			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                	n, 1, n, 1, recvG + i*n*n, n, w, 1, 0, vec_tmp, 1);
	 			daxpy(&n, &neg, recvG + s*n*n+i*n, &incx, vec_tmp, &incx);*/

				memset(vec, 0, sizeof(double)*n);
				tkn=(1+sqrt(1+4*tkp*tkp))/2.0;
	 			double al1 = (tkp-1.0)/tkn;
				double al2 = -(tkp-1.0)/tkn;
	//			double al1=(iter-2)/(iter+1);
	//		 	double al2=-(iter-2)/(iter+1);
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

				// do proximal:
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
				
	
			// Computing relative solution norm: [norm(wop-w)/norm(wop)]
			if(iter%freq==0){
				for (int z = 0; z < n; ++z)
				{	
					wsamp[z]=w[z];
				}
				daxpy(&n, &neg, wop, &incx, wsamp, &incx);
				resnrm=dnrm2(&n, wsamp, &incx)/dnrm2(&n, wop, &incx);
	//			if(rank==0){
//					cout<<resnrm<<endl;		
//				}
				if(resnrm <= tol){
					free(alpha); free(G); free(recvG);
					free(index); free(del_w); free(wsamp); free(sampres);
					sampcolidx.clear(); sampvals.clear();// sampres.clear();
					MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
					MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
					MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
					if(rank == 0){
						cout << "CA-BCD converged with residual: " << scientific << resnrm << setprecision(4) << fixed << " At iteration: " << iter << endl;
						cout << "Outer loop computation time: " << grammax*1000 << " ms" << endl;
						cout << "Inner loop computation time: " << innermax*1000 << " ms" << endl;
						cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << endl;
					}
					return;
			
				}
			}
		
		
	//printf("w after proximal at iter: %d\n",iter );
	//printme(w, n); 
			// Correct residual if any sampled row in current block appeared in any previous blocks
			/*for(int j = 0; j < i*b; ++j){
				for(int k = 0; k < b; ++k){
					if(index[j] == index[i*b + k])
						del_w[i*b + k] -= lambda*del_w[j];
				}
			}*/

			iter++;
			inneragg += MPI_Wtime() - innerst;
			if(iter == maxit){
				free(alpha); free(G); free(recvG);
				free(index); free(del_w); free(wsamp); free(sampres);
				//samprowidx.clear(); 
				sampcolidx.clear(); sampvals.clear(); //sampres.clear();
			 	MPI_Reduce(&gramagg, &grammax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			 	MPI_Reduce(&inneragg, &innermax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			 	MPI_Reduce(&commagg, &commmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			 	if(rank == 0){
			 		cout << "Outer loop computation time: " << grammax*1000 << " ms" << endl;
					cout << "Inner loop computation time: " << innermax*1000 << " ms" << endl;
			 		cout << "MPI_Allreduce time: " << commmax*1000 << " ms" << endl;
			 	}
//				myfile.close();
				return;
			}

		}
		
		//}
/*
		*/
		 
		

		


		gramst = MPI_Wtime();
		
		
		gramagg += MPI_Wtime() - gramst;

		
		/* Reset some buffers and start next outer iteration */

		//sampcolidx.clear(); 
		sampvals.clear(); //sampres.clear();
		//samprowidx[0] = 1;
		//memset(G, 0, sizeof(double)*gram_size*(gram_size+2));
		memset(G, 0, sizeof(double)*s*n*(n+1));
		memset(recvG, 0, sizeof(double)*s*n*(n+1));
		memset(del_w, 0, sizeof(double)*gram_size);
	}

}

//sampling tuning choices: randomly permute data matrix during I/O or after I/O. (This will perform better than current technique).
//
int main (int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	int npes, rank;
	int m, n;
	char *fname;
	char  *fnamewop;
	//myfile.open ("w.txt");
	double lambda, tol,b;
	int maxit, seed, freq, s;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm comm = MPI_COMM_WORLD;

	if(argc < 10)
	{
		if(rank == 0)
		{
			std::cout << "Bad args list!" << std::endl;
			std::cout << argv[0] << " [filename] [nrows] [ncols] [lambda] [maxit] [tol] [seed] [freq] [percent] [k] [number of benchmark iterations] [nnz]  [wop file] [initial step size] [Q] [beta]" << std::endl;
		}

		MPI_Finalize();
		return -1;
	}

	fname = argv[1];
	m = atoi(argv[2]);
	n = atoi(argv[3]);

	lambda = atof(argv[4]);
	maxit = atoi(argv[5]);
	tol = atof(argv[6]);
	seed = atoi(argv[7]);
	freq = atoi(argv[8]);
	b = atof(argv[9]);
	s = atoi(argv[10]);
	int niter = atoi(argv[11]);
	int nnz=atoi(argv[12]);
	fnamewop = argv[13];
	double gama_0 = atof(argv[14]);
	int Q = atoi(argv[15]);
	double beta = atof(argv[16]);

	if(rank==0){

	cout<<"filename: "<<argv[1]<<endl;
	cout<<"m: "<<argv[2]<<endl;
        cout<<"n: "<<argv[3]<<endl;
        cout<<"lambda: "<<argv[4]<<endl;
        cout<<"maxit: "<<argv[5]<<endl;
        cout<<"tol: "<<argv[6]<<endl;
        cout<<"seed: "<<argv[7]<<endl;
        cout<<"freq: "<<argv[8]<<endl;
        cout<<"b: "<<argv[9]<<endl;
	cout<<"s: "<<argv[10]<<endl;
        cout<<"niter: "<<argv[11]<<endl;
        cout<<"nnz: "<<argv[12]<<endl;
        cout<<"fnamewop: "<<argv[13]<<endl;
        cout<<"gama_0: "<<argv[14]<<endl;
        cout<<"Q: "<<argv[15]<<endl;
        cout<<"beta: "<<argv[16]<<endl;


	}
	//std::string lines = libsvmread(fname, m, n);

	//vector<int> rowidx, colidx;
	//vector<double> y, vals;
	//vector<int>::iterator it2;
	//vector<double>::iterator it1, it3;
	int *rowidx,*colidx;
	double *vals,*y;
	rowidx=(int *)malloc((m+1)*sizeof(int)); 
	colidx=(int *)malloc(nnz*sizeof(int)); 
	vals=(double *)malloc(nnz*sizeof(double)); 
	y=(double *)malloc(m*sizeof(double)); 

	int dual_method = 0;
	int nnz_pes=nnz/npes;
	if(rank==0){
		read(fname, rowidx, colidx, vals, y);
	}
	double *wop;
	wop=(double *)malloc(n*sizeof(double));
	string fnamestr(fnamewop);
	wopfile.open(fnamewop);

		for (int i = 0; i < n; ++i)
		{
			wopfile>>wop[i];
		}
//	cout<<"nnz:"<<rank;	
//cout<<"LLLLL"<<endl;
	// Load balancing and finding the indexed for scattering:
	int k=1;
	int pervi=0;
	int *rowidx_counts,*rowidx_disp,*colidx_counts,*colidx_disp,*val_counts,*val_disp,*y_counts,*y_disp;
	rowidx_counts=(int *)malloc(npes*sizeof(int));
	colidx_counts=(int *)malloc(npes*sizeof(int));
	val_counts=(int *)malloc(npes*sizeof(int));
	y_counts=(int *)malloc(npes*sizeof(int));
	rowidx_disp=(int *)malloc(npes*sizeof(int));
	colidx_disp=(int *)malloc(npes*sizeof(int));
	val_disp=(int *)malloc(npes*sizeof(int));
	y_disp=(int *)malloc(npes*sizeof(int));

	rowidx_disp[0]=0;
	colidx_disp[0]=0;
	val_disp[0]=0;
	y_disp[0]=0;
	//printf("nnz_pes= :%d\n",nnz_pes );
	if(rank==0){

	for (int i = 0; i < m+1; ++i)
	{
		if(rowidx[i]-1>=nnz_pes*k){
			
			rowidx_disp[k]=i;
			rowidx_counts[k-1]=i-pervi+1;
			colidx_counts[k-1]=rowidx[i]-rowidx[pervi];
			colidx_disp[k]=colidx_disp[k-1]+colidx_counts[k-1];
			val_counts[k-1]=rowidx[i]-rowidx[pervi];
			val_disp[k]=val_disp[k-1]+val_counts[k-1];
			y_counts[k-1]=rowidx_counts[k-1]-1;
			y_disp[k]=y_disp[k-1]+y_counts[k-1];
			pervi=i;
			k++;
			if(k==npes){
				rowidx_counts[k-1]=m-pervi+1;
				colidx_counts[k-1]=rowidx[m]-rowidx[pervi];
				val_counts[k-1]=rowidx[m]-rowidx[pervi];
				y_counts[k-1]=rowidx_counts[k-1]-1;
				break;
			}


		}
	}
	}
	MPI_Bcast(rowidx_counts,npes,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(colidx_counts,npes,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(y_counts,npes,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(val_counts,npes,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(rowidx_disp,npes,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(colidx_disp,npes,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(y_disp,npes,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(val_disp,npes,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	

	
int *local_rowidx,*local_colidx;
double *local_val,*local_y;
local_rowidx = (int *)malloc(rowidx_counts[rank]*sizeof(int)); 
local_colidx = (int *)malloc(colidx_counts[rank]*sizeof(int)); 
local_val = (double *)malloc(val_counts[rank]*sizeof(double)); 
local_y= (double *)malloc(y_counts[rank]*sizeof(double)); 



	if(strcmp(fname, "none") != 0){
		double scatterst = MPI_Wtime();
		MPI_Scatterv(rowidx, rowidx_counts, rowidx_disp, MPI_INT, local_rowidx, rowidx_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(colidx, colidx_counts, colidx_disp, MPI_INT, local_colidx, colidx_counts[rank], MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(vals, val_counts, val_disp, MPI_DOUBLE, local_val, val_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatterv(y, y_counts, y_disp, MPI_DOUBLE, local_y, y_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

		double scatterstp = MPI_Wtime();
		if(rank == 0){
			std::cout << "Finished Scatter of X and y in " << scatterstp - scatterst << " seconds." << std::endl;
			//free(X); free(y);
		}
	}
		

		int offset=local_rowidx[0]-1;
	for (int i = 0; i < rowidx_counts[rank]; ++i)
	{
		local_rowidx[i]=local_rowidx[i]-offset;
	}
	double algst, algstp;
	double *w;
	w = (double*) malloc (n*sizeof(double));
	//assert(0==Malloc_aligned(double, w, n, ALIGN));
	MPI_Barrier(MPI_COMM_WORLD);
	
	int Kmax=4;
	int Jmax=1;
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
			//cabcd(rowidx, colidx, vals, m, n, y, y.size(), lambda, s, b, maxit, tol, seed, freq, w, comm);
			algst = MPI_Wtime();
			for(int i = 0; i < niter; ++i){
				cabcd(local_rowidx, local_colidx, local_val, m, n, local_y,y_counts[rank],lambda, s, b, maxit,
				 tol, seed, freq, w, comm, wop, Q, beta, gama_0);
				//write_to_file(w, n);
				


	//		seed++;
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
//		printf("rank here is: %d\n",rank);	
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
	//myfile.close();

	/*if(rank == 0){
		std::cout << "w = ";
		for(int i = 0; i < n; ++i)
			std::cout << std::setprecision(4) << std::fixed << w[i] << " ";
		std::cout << std::endl;
	}*/
	
	/*
	free(localX); free(localy);
	free(cnts); free(displs);
	free(cnts2); free(displs2);
	*/
	MPI_Finalize();
	return 0;
}
