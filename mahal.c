/*
	Copyright (c) 2012 Jean-François Mousinho

	Permission is hereby granted, free of charge, to any person obtaining
	a copy of this software and associated documentation files (the
	"Software"), to deal in the Software without restriction, including
	without limitation the rights to use, copy, modify, merge, publish,
	distribute, sublicense, and/or sell copies of the Software, and to
	permit persons to whom the Software is furnished to do so, subject to
	the following conditions:

	The above copyright notice and this permission notice shall be
	included in all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
	EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
	MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
	NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
	LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
	OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
	WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	Description: tool of artimetic processement for supervisioned classifier
	using Mahalanobis distance and K-NN. 

	DEPENDENCIES

	gcc with libc and featuring C99.

	COMPILE

	$ make

	USAGE

	Check ./mahal -h

	EXAMPLES
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>

#ifndef uint
typedef unsigned int uint;
#endif

#ifndef MAKE_MVERBOSE
#define MAKE_MVERBOSE 0
#endif

#define VERBOSE if(verbose)
#define MATLAB if(matlab)
#define MVERBOSE if(MAKE_MVERBOSE)

#define BUFFER_INC 1024
#define BUFFER_SIZE 4096

typedef double cel_t;
#define CELFMT "%le"

typedef struct _matrix_t {
	uint N,M;
	cel_t *cel;
} matrix_t;

#define CEL(mat,LIN,COL) mat->cel[LIN*mat->M+COL]
#define CELR(mat,M,LIN,COL) mat[LIN*M+COL]

matrix_t *m_cof(matrix_t *A);
matrix_t *mahal_min(matrix_t *m,uint window);
matrix_t *mahal_knn(matrix_t *mTest,matrix_t *mTrain,matrix_t *inv,uint kParam);
matrix_t *m_trim_col(matrix_t **m,uint col_idx);

bool verbose = false;
bool matlab = false;
uint class_column = 0;
FILE *f_matlab = 0;

matrix_t *matrix_alloc(uint N,uint M,const char *tag)
{
	matrix_t *new_m = (matrix_t*)malloc(sizeof(matrix_t));

	new_m->N = N;
	new_m->M = M;
	new_m->cel = (cel_t*)malloc(N*M*sizeof(cel_t));

	MVERBOSE printf("MATRIX ALLOC (%s) %p\n",tag ? tag : "NA",(void*)new_m);

	return new_m;
}

void matrix_free(matrix_t *m,const char *tag)
{
	MVERBOSE printf("MATRIX FREE (%s) %p\n",tag ? tag : "NA",(void*)m);

	free(m->cel);
	free(m);
}

void print_m(matrix_t *A)
{
	uint i,j;

	for( i = 0 ; i < A->N ; i++ )
	{
		for( j = 0 ; j < A->M ; j++ )
			printf(CELFMT " ",CEL(A,i,j));

		printf("\n");
	}
}

/* matlab formated matrix printer */
void ml_print_m(const char *name,matrix_t *A)
{
	uint i,j;

	fprintf(f_matlab,"%s = [\n",name);

	for( i = 0 ; i < A->N ; i++ )
	{
		fprintf(f_matlab," [ ");
		for( j = 0 ; j < (A->M - 1) ; j++ )
			fprintf(f_matlab,CELFMT ", ",CEL(A,i,j));
		
		fprintf(f_matlab,CELFMT " ]; \n",CEL(A,i,j));
	}

	fprintf(f_matlab,"];\n");
}

/* calcula um vector de valores medios de determinada matriz */
matrix_t *m_mean(matrix_t *A)
{
	matrix_t *muA;
	uint i,j;

	muA = matrix_alloc(1,A->M,0);

	/* calcular vector de valores medios da matrix A */
	for( j = 0 ; j < A->M ; j++ )
	{
		/* para cada coluna... */
		cel_t sumCol;
		sumCol = 0.0;
		
		for( i = 0 ; i < A->N ; i++ )
			sumCol = sumCol + CEL(A,i,j);

		/* guardar valor no vector */
		CEL(muA,0,j) = sumCol / A->N;
	}

	return muA;
}

/* calcula a matriz da covariancia do vector ou matriz A */
matrix_t *m_cov(matrix_t *A)
{
	matrix_t *muA;
	uint i;
	matrix_t *cov;
	cel_t cov_ij;
	uint cov_i,cov_j;

	/* calcular matriz dos valores medios */
	muA = m_mean(A);

	/* alocar matriz de cov */
	cov = matrix_alloc(A->M,A->M,0);

	/* para cada celula da COV... */
	for( cov_i = 0 ; cov_i < A->M ; cov_i++ )
	{
		for( cov_j = 0 ; cov_j < A->M ; cov_j++ )
		{
			/* cada cov, cov_i e cov_j corresponde a uma feature.
			   somar em todas as observacoes o produto do (valor da feature
			   subtraido com a media dessa feature) com (valor da outra feature
			   subtraido com a media correspondente).
			    (feat_i - mean_feat_i ) * (feat_j - mean_feat_j) / N
			*/
			cov_ij = 0.0;
			for( i = 0 ; i < A->N ; i++ )
				cov_ij = cov_ij + 
					(CEL(A,i,cov_i) - CEL(muA,0,cov_i))*(CEL(A,i,cov_j) - CEL(muA,0,cov_j));
			cov_ij = cov_ij / (A->N - 1);

			/* guardar valor */
			CEL(cov,cov_i,cov_j) = cov_ij;
		}
	}

	/* returnar matriz, cliente e' responsavel por libertar memoria. */
	return cov;
}

/* calculo do produto de duas matrizes */
matrix_t *m_prod(matrix_t *A,matrix_t *B,matrix_t *res)
{
	uint i;
	uint res_i,res_j;
	cel_t sumL;

	assert( A->M == B->N );

	if( res == 0 ) {
		/* alocar matriz resultado */
		res = matrix_alloc(A->N,B->M,0);
	}

	for( res_i = 0; res_i < res->N ; res_i++ )
	{
		for( res_j = 0 ; res_j < res->M ; res_j++ )
		{
			/* SUM linha_i * col_j */
			sumL = 0.0;
			for( i = 0 ; i < B->N ; i++ )
				sumL = sumL + CEL(A,res_i,i)*CEL(B,i,res_j);
			CEL(res,res_i,res_j) = sumL;
		}
	}

	return res;
}

/* calculo da matriz minor de A, que consiste na matriz A sem
 a linha lin e sem a coluna col. resulta entao numa matriz de um
 grau em ambos os eixos menor. TESTED OK! */
matrix_t *m_minor(matrix_t *A,uint lin,uint col)
{
	uint i,j;
	uint res_i = 0,res_j = 0;
	matrix_t *res = matrix_alloc(A->N - 1, A->M - 1,0);

	for( i = 0 ; i < A->N ; i++ )
	{
		res_i = 0;

		for( j = 0 ; j < A->M ; j++ )
		{
			if( (i == lin) || (j == col) )
				continue;

			CEL(res,res_i,res_j) = CEL(A,i,j);
			res_j++;
		}

		if( i != lin )
			res_i++;
	}

	return res;
}

/* calculo do determinante de uma matriz A, TESTED OK! */
cel_t m_det(matrix_t *A)
{
	cel_t res;
	uint j;

	if( (A->N == A->M) && (A->M == 2) )
	{
		return CEL(A,0,0)*CEL(A,1,1) - CEL(A,1,0)*CEL(A,0,1);
	}

	if( (A->N == A->M) && (A->M == 3) )
	{
		/*
		  a1.b2.c3 - a1.b3.c2 - a2.b1.c3 + a2.b3.c1 + a3.b1.c2 - a3.b2.c1
		  
		  a1=CEL(A,0,0);
		  a2=CEL(A,0,1);
		  a3=CEL(A,0,2);

		  b1=CEL(A,1,0);
		  b2=CEL(A,1,1);
		  b3=CEL(A,1,2);

		  c1=CEL(A,2,0);
		  c2=CEL(A,2,1);
		  c3=CEL(A,2,2);

		  TESTED OK!
		*/
		return CEL(A,0,0)*CEL(A,1,1)*CEL(A,2,2) -
			CEL(A,0,0)*CEL(A,1,2)*CEL(A,2,1) -
			CEL(A,0,1)*CEL(A,1,0)*CEL(A,2,2) +
			CEL(A,0,1)*CEL(A,1,2)*CEL(A,2,0) +
			CEL(A,0,2)*CEL(A,1,0)*CEL(A,2,1) -
			CEL(A,0,2)*CEL(A,1,1)*CEL(A,2,0);
	}

	/*
	   det A = SUM aij Cij , (j ou i) = 0,...,k com matrix A_k*k

	   com Cij = (-1)^(i+j) Mij

	   Onde Mij = minor da matriz A, formada por eliminacao
	   da linha i e coluna j da matriz A.

	   Escolhendo manter a mesma linha i=0, fica:

	   det A = SUM a0j C0j C0j = SUM a0j * cof(0,j) , j=0...M.
	*/

	matrix_t *cof = m_cof(A);
	res = 0.0;
	for( j = 0 ; j < A->M ; j++ )
		res = res + CEL(A,0,j)*CEL(cof,0,j);
	matrix_free(cof,0);

	return res;
}

/* calculo da matriz transposta de A */
matrix_t *m_trans(matrix_t *A)
{
	uint i,j;
	matrix_t *res = matrix_alloc(A->M,A->N,0);

	for( i = 0 ; i < A->N ; i++ )
		for( j = 0 ; j < A->M ; j++ )
			CEL(res,j,i) = CEL(A,i,j);

	return res;
}

/* calculo da matriz dos cofactores de A, Cij = (-1)^(i+j) Mij, TESTED OK! */
matrix_t *m_cof(matrix_t *A)
{
	matrix_t *res,*minor;
	uint res_i,res_j;

	res = matrix_alloc(A->N,A->M,0);

	for( res_i = 0 ; res_i < A->N ; res_i++ )
	{
		for( res_j = 0 ; res_j < A->M ; res_j++ )
		{
			/* Cij = (-1)^(i+j) Mij */
			minor = m_minor(A,res_i,res_j);
			CEL(res,res_i,res_j) = (((res_i+res_j) & 1) ? -1.0 : 1.0) * m_det(minor);
			matrix_free(minor,0);
		}
	}

	return res;
}

/* devolve uma matriz cujos elementos foram todos multiplicados por s */
void m_scale(matrix_t *A,cel_t s,matrix_t *res)
{
	uint idx;
	for( idx = 0 ; idx < A->N * A->M ; idx++ )
		res->cel[idx] = A->cel[idx] * s;
}

/* calculo da matriz inversa de A. Usa metodo da matriz adjunta.
   inv A = 1/detA * Adj(A) = Trans(Cof_A)/det_A */
matrix_t *m_inv(matrix_t *A)
{
	matrix_t *cof,*coft;
	cel_t det;

	/* se a matriz for 2x2.. usar atalho */
	if( (A->N == 2) && (A->M == 2) )
	{
		det = m_det(A);
		matrix_t *res = matrix_alloc(2,2,0);
		CEL(res,0,0) = CEL(A,1,1);
		CEL(res,0,1) = -1.0 * CEL(A,0,1);
		CEL(res,1,0) = -1.0 * CEL(A,1,0);
		CEL(res,1,1) = CEL(A,0,0);
		m_scale(res,1.0/det,res);
		return res;
	}

	cof = m_cof(A);
	coft = m_trans(cof);
	matrix_free(cof,0);
	det = m_det(A);

	m_scale(coft,1.0/det,coft);

	return coft;
}

/* duplica matriz, memoria e' alocada para a nova matriz */
matrix_t *m_dup(matrix_t* A)
{
	matrix_t *res;

	res = matrix_alloc(A->N,A->M,0);
	memcpy(res->cel,A->cel,sizeof(cel_t)*A->N*A->M);

	return res;
}

/* carrega ficheiro de matriz para tipo matrix_t */
matrix_t *m_load(char *file)
{
	matrix_t *res;
	FILE *f;

	/* open file */
	f = fopen(file,"r");
	if( !f )
		return 0;

	/* primeiro obter as dimensioes da matriz lendo o ficheiro */
	uint N=0,M=0;
	char *buffer_ptr = 0;
	size_t buffer_size = BUFFER_SIZE;
	char *aux_ptr;
	cel_t aux_double;
	char *fop;

	buffer_ptr = (char*)malloc(buffer_size);

	fop = fgets(buffer_ptr,buffer_size,f);
	if( fop == NULL )
	{
		fprintf(stderr,"mahal: error reading from file %s, aborting!\n",file);
		free(buffer_ptr);
		fclose(f);
		return 0;
	}

	/* determinar grau do vector */
	for( aux_ptr = strtok(buffer_ptr," \t\r\n") ; 
			aux_ptr ; 
			aux_ptr = strtok(NULL," \t\r\n") )
	{
		if( sscanf(aux_ptr,CELFMT,&aux_double) == 1 )
			M++;
	}

	/* agora que mediu-se M, vamos medir N */
	
	N = 0;
	rewind(f);
	while( !feof(f) )
	{
		fop = fgets(buffer_ptr,buffer_size,f);
		
		if( strchr(buffer_ptr,'\n') )
			*strchr(buffer_ptr,'\n') = '\0';

		if( strchr(buffer_ptr,'\r') )
			*strchr(buffer_ptr,'\r') = '\0';

		if( !strlen(buffer_ptr) )
			continue;

		if( fop == NULL ) {
			break;
		}

		if( (fop == NULL) && (errno == EFAULT) ) {
			fprintf(stderr,"mahal: error reading from file %s, aborting!\n",file);
			free(buffer_ptr);
			fclose(f);
			return 0;
		}
		N++;
	}

	VERBOSE printf("mahal: detected matrix %ux%u from file %s\n",N,M,file); 

	/* agora temos as dimensioes, alocar a matriz */
	res = matrix_alloc(N,M,0);

	/* ler matriz para estrutura de dados */
	rewind(f);
	uint i=0,j=0;
	while(!feof(f))
	{
		clearerr(f);
		fop = fgets(buffer_ptr,buffer_size,f);
		if(ferror(f)) break;
		if( !fop ) break;
		
		/* obter vector */
		j = 0;
		for( aux_ptr = strtok(buffer_ptr," \t\r\n") ;
				aux_ptr ;
				aux_ptr = strtok(NULL," \t\r\n") )
		{
			if( !sscanf(aux_ptr,CELFMT,&aux_double) )
				continue;

			/* temos coordenada j */
			if( j >= M ) {
				fprintf(stderr,"mahal: non-uniform vector size in file %s (near line %u), aborting!\n",file,i);
				goto return_fail;
			}

			/* verificar se indice de vector esta dentro dos limites */
			if( i >= N ) {
				fprintf(stderr,"mahal: unexpected difference of number of vectors, aborting!\n");
				goto return_fail;
			}

			/* guardar valor do vector */
			CEL(res,i,j) = aux_double;

			/* incrementar indice de coordenada */
			j++;
		}

		/* chegou ao fim do vector, incrementar indice de vector */
		i++;
	}

	/* leu ficheiro com sucesso! */
	VERBOSE printf("mahal: read matrix file %s successful (%u lines and %u columns)\n",file,N,M);
	free(buffer_ptr);
	fclose(f);
	return res;

return_fail:
	matrix_free(res,0);
	free(buffer_ptr);
	fclose(f);
	return 0;
}

cel_t m_min(matrix_t *m)
{
	uint i;
	cel_t min = sqrt(-1.0);
	
	if( m->N == 1 )
   	{
		/* single row */

		for( i = 0 ; i < m->M ; i++ )
		{
			if( isnan(min) ) {
				min = CEL(m,0,i);
				continue;
			}

			if( min <= CEL(m,0,i) )
				continue;

			min = CEL(m,0,i);
		}
	}

	if( m->M == 1 )
	{
		/* single column */

		for( i = 0 ; i < m->N ; i++ )
		{
			if( isnan(min) ) {
				min = CEL(m,i,0);
				continue;
			}

			if( min <= CEL(m,i,0) )
				continue;

			min = CEL(m,i,0);
		}
	}

	return min;
}

cel_t m_max(matrix_t *m)
{
	uint i;
	cel_t max = sqrt(-1.0);
	
	if( m->N == 1 )
   	{
		/* single row */

		for( i = 0 ; i < m->M ; i++ )
		{
			if( isnan(max) ) {
				max = CEL(m,0,i);
				continue;
			}

			if( max > CEL(m,0,i) )
				continue;

			max = CEL(m,0,i);
		}
	}

	if( m->M == 1 )
	{
		/* single column */

		for( i = 0 ; i < m->N ; i++ )
		{
			if( isnan(max) ) {
				max = CEL(m,i,0);
				continue;
			}

			if( max > CEL(m,i,0) )
				continue;

			max = CEL(m,i,0);
		}
	}

	return max;
}

void usage(void)
{
	printf("mahal v0.1 "
			"Copyright (C) 2010 Jean-François Mousinho\n"
			"Instituto Superior Tecnico - Universidade Tecnica de Lisboa\n"
			"\n"
			"Usage: mahal [options] [-t file] [-i file] [-o file] [-k kParam]\n"
			"     -t            Train set matrix file, def: set_train.txt\n"
			"     -i            Test set matrix file, def: set_test.txt\n"
			"     -o            Output results file, def: stdout\n"
			"Options:\n"
			"     -m            Matlab output (useful for results checks)\n"
			"     -e            Apply Euclidean distance\n"
			"     -k <k-Param>  Apply K-NN with K-Param and Mahalanobis distance\n"
			"     -c <col-idx>  Column index where the class is (one-based)\n"
			"     -v            Verbose mode.\n\n");
	exit(EXIT_SUCCESS);
}

int main(int argc,char *argv[])
{
	matrix_t *muA;
	matrix_t *mTest,*mTestClass = 0;
	matrix_t *mTrain,*mTrainClass = 0;
	cel_t dTrainClassMax = 0,dTrainClassMin = 0;
	cel_t dTestClassMax,dTestClassMin;
	matrix_t *res;
	uint i,j,aux_idx;
	char setTestFile[FILENAME_MAX] = {"set_test.txt"};
	char setTrainFile[FILENAME_MAX] = {"set_train.txt"};
	char matlabOutput[FILENAME_MAX] = {0};
	char outputFile[FILENAME_MAX] = {0};
	uint kParam = 0;
	FILE *fout;
	bool euclidean;
	matrix_t *xy_kdist;
	cel_t classAvg;
	uint truePositiveCount = 0,falsePositiveCount = 0;
	uint trueNegativeCount = 0,falseNegativeCount = 0;

	char ch;
	while ((ch = getopt(argc, argv, "vc:m:ek:i:t:o:h")) != -1) {
		switch(ch)
		{
			case 'v':
				verbose = true;
				break;
			case 'c':
				if( !sscanf(optarg,"%u",&class_column) )
					usage();
				break;
			case 'k':
				if( !sscanf(optarg,"%u",&kParam) )
					usage();
				break;
			case 'e':
				euclidean = true;
				break;
			case 'i':
				strcpy(setTestFile,optarg);
				break;
			case 't':
				strcpy(setTrainFile,optarg);
				break;
			case 'o':
				strcpy(outputFile,optarg);
				break;
			case 'm':
				strcpy(matlabOutput,optarg);
				matlab = true;
				break;
			case 'h':
			case '?':
			default:
				usage();
		}
	}

	if( matlab )
	{
		f_matlab = fopen(matlabOutput,"w");
		if( !f_matlab ) {
			fprintf(stderr,"mahal: failed to open %s file for writting MatLab output\n",matlabOutput);
			return EXIT_FAILURE;
		}
	}

	mTest = m_load(setTestFile);
	mTrain = m_load(setTrainFile);

	MATLAB fprintf(f_matlab,"clear all\nclose all\n");
	MATLAB ml_print_m("mTest",mTest);
	MATLAB ml_print_m("mTrain",mTrain);

	if( class_column ) 
	{
		/* extract the class column from the matrixes */
		mTrainClass = m_trim_col(&mTrain,class_column - 1);

		dTrainClassMax = m_max(mTrainClass);
		dTrainClassMin = m_min(mTrainClass);

		VERBOSE printf("mahal: extracted class column from train set (min=" CELFMT ", max=" CELFMT ").\n",
				dTrainClassMin, dTrainClassMax);

		MATLAB fprintf(f_matlab,"trainClass = mTrain(:,%u);\n",class_column);
		MATLAB fprintf(f_matlab,"mTrain(:,%u) = [];\n",class_column);

		if( (mTest->M - 1) == mTrain->M ) {
			/* test set also has the class column */
			mTestClass = m_trim_col(&mTest,class_column - 1);
			dTestClassMax = m_max(mTestClass);
			dTestClassMin = m_min(mTestClass);
			VERBOSE printf("mahal: extracted class column from test set (min=" CELFMT ", max=" CELFMT ").\n",
					dTestClassMin,dTestClassMax);
		}
	}

	matrix_t *cov = m_cov(mTrain);
	VERBOSE printf("cov(setTrain):\n");
	VERBOSE print_m(cov);
	VERBOSE printf("\n");
	MATLAB ml_print_m("mS",cov);

	matrix_t *inv = m_inv(cov);
	VERBOSE printf("cov(setTrain)^-1:\n");
	VERBOSE print_m(inv);
	VERBOSE printf("\n");
	MATLAB ml_print_m("mInvS",inv);
	MATLAB fprintf(f_matlab,
			"if isequal(mInvS,inv(cov(mTrain)))\n"
			"   fprintf(1,'mInvS OK\\n');\n"
			"else\n"
			"   fprintf(1,'mInvS NOT OK!\\n');\n"
			"end\n" );

	muA = m_mean(mTrain);
	VERBOSE printf("mean setTrain:\n");
	VERBOSE print_m(muA);
	VERBOSE printf("\n");
	MATLAB ml_print_m("mAvgTrain",muA);
	MATLAB fprintf(f_matlab,
			"if isequal(mAvgTrain,mean(mTrain))\n"
			"   fprintf(1,'mAvgTrain OK\\n');\n"
			"else\n"
			"   fprintf(1,'mAvgTrain NOT OK!\\n');\n"
			"end\n" );

	if( !kParam )
	{
		/*
		   d(I) = (Y(I,:) - mu) x Inv(S) x (Y(I,:) - mu)

		   -- mathworks.com
		*/

		res = matrix_alloc(mTest->N,mTrain->M,"res");
		for( i = 0 ; i < mTest->N ; i++ )
		{
			for( j = 0 ; j < mTrain->M ; j++ )
				CEL(res,i,j) = CEL(mTest,i,j) - CEL(muA,0,j);
		}

		VERBOSE printf("Y(I,:) - mu\n");
		VERBOSE print_m(res);
		VERBOSE printf("\n");

		matrix_t *yiMmu = m_dup(res);
		matrix_t *aux1,*aux2;

		/* transpose(yiMmu) */
		aux2 = m_trans(res);
		matrix_free(res,"res");

		VERBOSE printf("(Y(I,:) - mu)'\n");
		VERBOSE print_m(aux2);
		VERBOSE printf("\n");

		/* Inv(S) * yiMmu */
		aux1 = m_prod(inv,aux2,0);
		matrix_free(aux2,"aux2");

		VERBOSE printf("inv*(Y(I,:) - mu)'\n");
		VERBOSE print_m(aux1);
		VERBOSE printf("\n");

		
		/* yiMmu * aux */
		aux2 = m_prod(yiMmu,aux1,0);
		matrix_free(yiMmu,"yiMmu");
		matrix_free(aux1,"aux1");

		VERBOSE printf("(Y(I,:) - mu) * ...\n");
		VERBOSE print_m(aux2);
		VERBOSE printf("\n");

		if( outputFile[0] == '\0' )
			fout = stdout;
		else
			fout = fopen(outputFile,"w");

		if( !fout ) {
			fprintf(stderr,"mahal: unable to open file %s for writting!\n",outputFile);
			matrix_free(aux2,"aux2");
			goto return_fail;
		}

		for( i = 0 ; i < aux2->N ; i++ )
		   fprintf(fout,CELFMT "\n",CEL(aux2,i,i));

		if( outputFile[0] != '\0' )
			fclose(fout);

		matrix_free(aux2,"aux2");

	} else
	{
		/* 
		   Apply K-NN classifier with k-param and use Mahalanobis distance.
		   
		   For each point in Y (l x c), yi we do (yi-xi) * S^-1 * (yi-xi) instead
		   of (yi-mu) * S^-1 (yi-mu), this means that for each test point
		   we'll need Y.l rows. resulting matrix will
		   have the same number of rows of Y. */

		if( outputFile[0] == '\0' )
			fout = stdout;
		else
			fout = fopen(outputFile,"w");

		if( !fout ) {
			fprintf(stderr,"mahal: unable to open file %s for writting!\n",outputFile);
			goto return_fail;
		}


		/* Allocate the distance matrix, first column is the index of the nearest train
		   vector based on K-NN, second column is the resulting distance value.
		   The first vector of m_dist corresponds to the K-NN applied to the first
		   vector of the test set. */
		
		xy_kdist = mahal_knn(mTest,mTrain,inv,kParam);

		if( class_column )
		{
			/* replace train vector indexes with class values */	
			for( i = 0 ; i < mTest->N ; i++ )
			{
				for( j = 0 ; j < kParam ; j++ ) {
					aux_idx = CEL(xy_kdist,i,j) - 1;
					CEL(xy_kdist,i,j) = CEL(mTrainClass,aux_idx,0);
				}
			}

			if( mTestClass && ((dTrainClassMax - dTrainClassMin) == 1) )
			{
				/* if we've mTestClass, we're testing the train set, provide the results */
		
				for( i = 0 ; i < mTest->N ; i++ )
				{
					for( j = 0, classAvg = 0 ; j < kParam ; j++ ) {
						/* average the class value */
						classAvg += CEL(xy_kdist,i,j);
					}
					if( classAvg > ((double)kParam / 2.0) ) {
						/* result of KNN is 1 */
						if( CEL(mTestClass,i,0) == 1.0 )
							truePositiveCount++;
						else
							falsePositiveCount++;
					} else
					{
						/* result of KNN is 0 */
						if( CEL(mTestClass,i,0) == 0.0 )
							trueNegativeCount++;
						else
							falseNegativeCount++;
					}
				}

				/* end of test-train stats */
			}
		}

		for( i = 0 ; i < xy_kdist->N ; i++ )
		{
			for( j = 0 ; j < xy_kdist->M ; j++ )
				printf("%0.0lf ",CEL(xy_kdist,i,j));

			printf("\n");
		}
		//print_m(xy_kdist);
		MATLAB ml_print_m("xy_kdist",xy_kdist);

		if( mTestClass && ((dTrainClassMax - dTrainClassMin) == 1) ) {
			printf(	"truePositiveCount = %u (%0.2lf%%)\n"
					"trueNegativeCount = %u (%0.2lf%%)\n",
					truePositiveCount, (double) truePositiveCount / (truePositiveCount + falsePositiveCount) * 100.0,
					trueNegativeCount, (double) trueNegativeCount / (trueNegativeCount + falseNegativeCount) * 100.0);
		}

		matrix_free(xy_kdist,"xy_kdist");
		if( outputFile[0] != '\0' )
			fclose(fout);
	}

	return EXIT_SUCCESS;

return_fail:
	return EXIT_FAILURE;
}

matrix_t *dist_NN_to_N(matrix_t *m)
{
	matrix_t *res = matrix_alloc(1,m->N,0);

	assert(m->N == m->M);

	for( uint i = 0 ; i < m->N ; i++ )
		CEL(res,0,i) = CEL(m,i,i);

	return res;
}

matrix_t *mahal_knn(matrix_t *mTest,matrix_t *mTrain,matrix_t *inv,uint kParam)
{
	uint i,j,k;
	matrix_t *m_difL, *m_difR, *m_difR_T, *m_SpDRT;
	matrix_t *res,*dist_all,*kmin_dist;
	matrix_t *dist_allN;

	/* allocate final result matrix, indexes of the nearest kParam points relative
	   to the mTrain matrix. */
	res = matrix_alloc(mTest->N,kParam,"res");

	MATLAB fprintf(f_matlab,"xy_distML = [];\n");

	for( i = 0 ; i < mTest->N ; i++ )
	{
		m_difL = matrix_alloc(mTrain->N,mTest->M,"m_difL");
		for( j = 0 ; j < mTrain->N ; j++ )
		{
			for( k = 0 ; k < mTrain->M ; k++ )
				CEL(m_difL,j,k) = CEL(mTest,i,k) - CEL(mTrain,j,k);
		}

		VERBOSE printf("Y(I,:) - X(I,:)\n");
		VERBOSE print_m(m_difL);
		VERBOSE printf("\n");

		m_difR =  m_dup(m_difL);

		/* transpose(yiMmu) */
		m_difR_T = m_trans(m_difR);
		matrix_free(m_difR,"m_difR");

		VERBOSE printf("(Y(I,:) - X(I,:))'\n");
		VERBOSE print_m(m_difR_T);
		VERBOSE printf("\n");

		/* Inv(S) * yiMmu */
		m_SpDRT = m_prod(inv,m_difR_T,0);
		matrix_free(m_difR_T,"m_difR_T");

		VERBOSE printf("inv*(Y(I,:) - X(I,:))'  [%u x %u]\n",m_SpDRT->N,m_SpDRT->M);
		VERBOSE print_m(m_SpDRT);
		VERBOSE printf("\n");


		/* yiMmu * aux */
		dist_all = m_prod(m_difL,m_SpDRT,0);
		matrix_free(m_difL,"m_difL");
		matrix_free(m_SpDRT,"m_SpDRT");

		/* get N*N matrix diagonal into a single row */
		dist_allN = dist_NN_to_N(dist_all);
		matrix_free(dist_all,"dist_all");

		VERBOSE printf("(Y(I,:) - X(I,:)) * ...\n");
		VERBOSE print_m(dist_allN);
		VERBOSE printf("\n");
		MATLAB ml_print_m("distAll",dist_allN);

		/* Call mahal_min function which will sort all distances in res matrix
		   and return the first kParam vectors corresponding to the kParam
		   nearest vectors to the test point xi. The returning matrix has kParam
		   rows and one column corresponding to the vector index in res. */
		kmin_dist = mahal_min(dist_allN,kParam);

		/* in matlab indexes are one-based, do kmin_dist(i,1)++ */
		for( j = 0 ; j < kParam ; j++ )
			CEL(kmin_dist,j,0) = CEL(kmin_dist,j,0) + 1;

		/* print matlab matrix */
		MATLAB ml_print_m("minKDist",kmin_dist);

		/* copy information into result matrix */
		for( j = 0 ; j < kParam ; j++ )
			CEL(res,i,j) = CEL(kmin_dist,j,0);
		
		matrix_free(dist_allN,"dist_allN");
		matrix_free(kmin_dist,"kmin_dist");

		MATLAB {
			fprintf(f_matlab,
			   "\n"
			   "  v = mTest(%u,:);\n"
			   "  mXY = ones(length(mTrain),1)*v - mTrain;\n"
			   "  distAllNN = mXY*mInvS*(mXY');\n"
			   "  res = zeros(2,length(distAllNN));\n"
			   "  for i=1:length(distAllNN)\n"
			   "    res(1,i) = i;\n"
			   "    res(2,i) = distAllNN(i,i);\n"
			   "  end\n"
			   "  res = transpose(res);\n"
			   "  res_sorted = sortrows(res,2);\n"
			   "  res_sorted(6:end,:) = [];\n"
			   "  res_sorted(:,2) = [];\n"
			   "  xy_distML = [ xy_distML ; transpose(res_sorted) ];\n"
			   "\n",i+1);

			if( mTrain->M == 2 ) {
				fprintf(f_matlab,
						"figure\n"
						"scatter(mTrain(:,1),mTrain(:,2),20,distAll);\n"
						"hold on\n"
						"hb = colorbar;\n"
						"ylabel(hb,'Mahalanobis Distance')\n"
						"plot(mTest(%u,1),mTest(%u,2),'xr');\n"
						"for k=1:%u\n"
						"  plot(mTrain(minKDist(k,1),1),mTrain(minKDist(k,1),2),'.r');\n"
						"end\n"
						"hold off\n",
						i+1,i+1,kParam);
			}
		}
	}
	return res;
}

matrix_t *mahal_min(matrix_t *m,uint window)
{
	uint i,j,k;
	cel_t aux_d;

	matrix_t *res = matrix_alloc(window,2,0);

	for( i = 0 ; i < window ; i++ ) {
		CEL(res,i,1) = -1.0;
		CEL(res,i,0) = 0;
	}

	for( i = 0 ; i < m->M ; i++ )
	{
		aux_d = CEL(m,0,i);

		/* lookup right place in res, order ascending */
		for( j = 0 ; j < window ; j++ )
		{
			if( CEL(res,j,1) == -1.0 ) {
				CEL(res,j,0) = (cel_t) i;
				CEL(res,j,1) = aux_d;
				break;
			}

			/* if current distance is below current window entry,
			   replace the window entry with current distance and
			   stop looking a place to store it. */
			if( aux_d > CEL(res,j,1) ) 
				continue;

			/* shift all entries in the window one position up */
			for( k = (window - 1) ; k > j ; k-- ) {
				CEL(res,k,0) = CEL(res,(k-1),0);
				CEL(res,k,1) = CEL(res,(k-1),1);
			}

			CEL(res,j,0) = (cel_t) i;
			CEL(res,j,1) = aux_d;
			break;
		}
	}

	return res;
}

/* removes a single column from the matrix m and returns it to the caller. */
matrix_t *m_trim_col(matrix_t **m,uint col_idx)
{
	matrix_t *m_col = matrix_alloc((*m)->N,1,"m_col");
	matrix_t *m_new = matrix_alloc((*m)->N,(*m)->M-1,"m_new");
	uint i,j,k;

	for( i = 0 ; i < (*m)->N ; i++ )
	{
		/* for each line, copy all columns except col_idx to m_new */
		for( j = 0, k = 0 ; j < (*m)->M ; j++ )
		{
			if( j == col_idx) {
				CEL(m_col,i,0) = CEL((*m),i,j);
				continue;
			}

			CEL(m_new,i,k++) = CEL((*m),i,j);
		}
	}

	/* we're done, free old matrix in m, replace with m_new and return m_col */
	matrix_free(*m,0);
	*m = m_new;

	return m_col;
}

