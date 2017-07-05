#include "mex.h"

#include <cmath>
#include <cstring>
#include <random>
//#include <iostream>

std::default_random_engine generator;
std::normal_distribution<double>        normal_distribution(0., 1.);
std::uniform_real_distribution<double>  unif_distribution(0.0, 1.0);

#define datatype double /* type of the elements in y */


double compute_square_norm2(const double *y, const double *H, const double *x, int M, int N);

void compute_grad(const double *y, const double *H, const double *x, double *grad, double sigma2, int M, int N);

double normq2(const double *x, int N);

double norminf(const double *x, int N);

double distl2(const double *x1, const double *x2, int N);

void proxCondat(const double *y, double *x, double a, int N);

static void l1ballproj_Condat(datatype* y, datatype* x,
    const unsigned int length, const double a);



void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
        int nrhs, const mxArray *prhs[]) /* Input variables  */
{
    
    if (nrhs != 16)
        mexErrMsgTxt("Wrong number of input arguments.");

    if (nlhs != 3)
        mexErrMsgTxt("Wrong number of output arguments.");    
    
        // inputs
    #define Y_IN                prhs[0]
    #define H_IN                prhs[1]
    #define NIT_IN              prhs[2]
    #define X_IN                prhs[3]
    #define MU_IN               prhs[4]
    #define SIGMA2_IN           prhs[5]    
    #define SAMPLE_X_IN         prhs[6]
    #define SAMPLE_MU_IN        prhs[7]
    #define SAMPLE_SIGMA2_IN    prhs[8]
    #define A_MU_IN             prhs[9]
    #define B_MU_IN             prhs[10]
    #define A_SIGMA2_IN         prhs[11]
    #define B_SIGMA2_IN         prhs[12]
    #define MAXEIGHH_IN         prhs[13]
    #define THINNING_IN         prhs[14]
    #define SEED_IN             prhs[13]
    
        // Outputs
    #define CHAIN_X_OUT         plhs[0]
    #define CHAIN_MU_OUT        plhs[1]
    #define CHAIN_SIGMA2_OUT    plhs[2]
    
    double *y, *H, *x_init, *chain_x, *chain_sigma2, *chain_mu;
    int Nit;
    bool sample_x, sample_mu, sample_sigma2;
    int N, M;
    double mu, sigma2, beta, a_mu, b_mu, a_sigma2, b_sigma2;
    int thinning;
    double maxEigHH, seed;
    
    int i, j, n_it;
    double current_error2;
    double Lf, gamma_myula, lambda_myula;
    
    y = mxGetPr(Y_IN);
    H = mxGetPr(H_IN);
    Nit = mxGetScalar(NIT_IN);
    
    x_init = mxGetPr(X_IN);
    mu     = mxGetScalar(MU_IN);
    sigma2 = mxGetScalar(SIGMA2_IN);
    
    sample_x      = mxGetScalar(SAMPLE_X_IN);
    sample_mu     = mxGetScalar(SAMPLE_MU_IN);
    sample_sigma2 = mxGetScalar(SAMPLE_SIGMA2_IN);
    
    a_mu = mxGetScalar(A_MU_IN);
    b_mu = mxGetScalar(B_MU_IN);
    a_sigma2 = mxGetScalar(A_SIGMA2_IN);
    b_sigma2 = mxGetScalar(B_SIGMA2_IN);
    
    M = mxGetM(H_IN);
    N = mxGetN(H_IN);
    beta = 2. * mu * (double)N * sigma2;
    
    maxEigHH   = mxGetScalar(MAXEIGHH_IN);
    thinning = mxGetScalar(THINNING_IN);

    seed     = mxGetScalar(SEED_IN);
    generator.seed(seed);    
    

    double g1_grad[N];
    double x_prox[N];
    
     /* Create the output matrix */
    if (sample_x)
        CHAIN_X_OUT     = mxCreateDoubleMatrix(N, Nit, mxREAL);
    else
        CHAIN_X_OUT     = mxCreateDoubleMatrix(N, 1, mxREAL);

    
    if (sample_mu)
        CHAIN_MU_OUT     = mxCreateDoubleMatrix(1, Nit, mxREAL);
    else
        CHAIN_MU_OUT     = mxCreateDoubleMatrix(1, 1, mxREAL);
    
    
    if (sample_sigma2)
        CHAIN_SIGMA2_OUT     = mxCreateDoubleMatrix(1, Nit, mxREAL);
    else
        CHAIN_SIGMA2_OUT     = mxCreateDoubleMatrix(1, 1, mxREAL);
    
    
    chain_x      = mxGetPr(CHAIN_X_OUT);
    chain_mu     = mxGetPr(CHAIN_MU_OUT);
    chain_sigma2 = mxGetPr(CHAIN_SIGMA2_OUT);

    
    if (!sample_mu)
        chain_mu[0] = mu;
    
    if (!sample_sigma2)
        chain_sigma2[0] = sigma2;
    
    double x[N];
    for (j = 0; j < N; j++) {
        x[j] = x_init[j];
        
        if (!sample_x)
            chain_x[j] = x_init[j];
    }
    
    
    /**    *****************    **/
    /*       Starting loop       */
    /**    *****************    **/

    current_error2 = compute_square_norm2(y, H, x, M, N);
    
    for(n_it = 0; n_it < Nit; n_it++) {

        /** Sample x **/
        if (sample_x) {
            
            Lf = maxEigHH / sigma2;
            lambda_myula = 1. / Lf;
            gamma_myula  = 1. / (4. * Lf);
            
            // Loop defined by thinning
            for (int th = 0; th < thinning; th++) {
                
                // Compute gradient
                compute_grad(y, H, x, g1_grad, sigma2, M, N);
                
                // Compute prox
                proxCondat(x, x_prox, lambda_myula * mu * (double)N, N);
                
                // Update
                for(j = 0; j < N; j++) {
                    x[j] = (1. - gamma_myula / lambda_myula) * x[j]
                            - gamma_myula * g1_grad[j]
                            + gamma_myula / lambda_myula * x_prox[j]
                            + sqrt(2. * gamma_myula) * normal_distribution(generator);
                }         
                
            }
            
            current_error2 = compute_square_norm2(y, H, x, M, N);
            
            // Save x
            for (j=0; j < N; j++)
                chain_x[j + n_it * N] = x[j];
        }
        
        
        
        
        /** Sample mu **/
        if (sample_mu) {
            
            double a_post = a_mu + (double)N;
            double b_post = 1 / (b_mu + (double)N * norminf(x, N));
            
            std::gamma_distribution<double> gamma_distribution(a_post, b_post);
            
            mu = gamma_distribution(generator);
            
            // Save mu
            chain_mu[n_it] = mu; 
        }
        
        
        
        
        
        /** Sample sigma **/
        if (sample_sigma2) {
            
            double a1 = a_sigma2 + (double)M / 2;
            double b2 = 1. / (b_sigma2 + 0.5 * current_error2);
            std::gamma_distribution<double> gamma_distribution(a1, b2);
            
            sigma2 = 1. / gamma_distribution(generator);

            // Save sigma2
            chain_sigma2[n_it] = sigma2;
        }
    }
    
    return;
}








/* ********************************************************************* */
/*                                                                       */
/*                          Fonctions annexes                            */
/*                                                                       */
/* ********************************************************************* */

double compute_square_norm2(const double *y, const double *H, const double *x, int M, int N) {
    
    double norm = 0;
    double buf = 0;
    int i, j;
    for (i = 0; i < M; i++) {
        buf = y[i];
        
        for (j = 0; j < N; j++)
            buf -= H[i + j * M] * x[j];
        
        norm += buf * buf;
    }
    return norm;
}


void compute_grad(const double *y, const double *H, const double *x, double *grad, double sigma2, int M, int N) {
    
    double buf[M];
    
    // Compute buf = Hx - y
    for (int i = 0; i < M; i++) {
        
        buf[i] = - y[i];
        for (int j = 0; j < N; j++)
            buf[i] +=  H[i + j * M] * x[j];
    }
    
    // Compute grad = (H' * buf) / sigma2
    for (int j = 0; j < N; j++) {
        
        grad[j] = 0;
        for (int i = 0; i < M; i++)
            grad[j] +=  H[i + j * M] * buf[i];
        
        grad[j] /= sigma2;
    }
}


double normq2(const double *x, int N) {
    
    double output = 0;
    
    for (int j = 0; j < N; j++) {
        double buf = x[j];
        output += buf * buf;
    }
            
    return output;
}


double norminf(const double *x, int N) {
    
    double output = std::abs(x[0]);
    
    for (int j = 0; j < N; j++)
        if (std::abs(x[j]) > output)
            output = std::abs(x[j]);
            
    return output;
}

double distl2(const double *x1, const double *x2, int N) {
    
    double buf[N];
    for (int j = 0; j < N; j++)
        buf[j] = x1[j] - x2[j];
    
    return normq2(buf, N);
}

void proxCondat(const double *y, double *x, double a, int N) {

	double buf[N];

    for (int j = 0; j < N; j++)
        buf[j] = y[j] / a;
    
	l1ballproj_Condat(buf, x, N, 1);
    
    for (int j = 0; j < N; j++)
        x[j] = y[j] - a * x[j];
}


static void l1ballproj_Condat(datatype* y, datatype* x,
const unsigned int length, const double a) {
	if (a<=0.0) {
		if (a==0.0) std::memset(x,0,length*sizeof(datatype));
		return;
	}
	datatype*	aux = (x==y ? (datatype*)malloc(length*sizeof(datatype)) : x);
	int		auxlength=1;
	int		auxlengthold=-1;
	double	tau=(*aux=(*y>=0.0 ? *y : -*y))-a;
	int 	i=1;
	for (; i<length; i++)
		if (y[i]>0.0) {
			if (y[i]>tau) {
				if ((tau+=((aux[auxlength]=y[i])-tau)/(auxlength-auxlengthold))
				<=y[i]-a) {
					tau=y[i]-a;
					auxlengthold=auxlength-1;
				}
				auxlength++;
			}
		} else if (y[i]!=0.0) {
			if (-y[i]>tau) {
				if ((tau+=((aux[auxlength]=-y[i])-tau)/(auxlength-auxlengthold))
				<=aux[auxlength]-a) {
					tau=aux[auxlength]-a;
					auxlengthold=auxlength-1;
				}
				auxlength++;
			}
		}
	if (tau<=0) {	/* y is in the l1 ball => x=y */
		if (x!=y) std::memcpy(x,y,length*sizeof(datatype));
		else free(aux);
	} else {
		datatype*  aux0=aux;
		if (auxlengthold>=0) {
			auxlength-=++auxlengthold;
			aux+=auxlengthold;
			while (--auxlengthold>=0)
				if (aux0[auxlengthold]>tau)
					tau+=((*(--aux)=aux0[auxlengthold])-tau)/(++auxlength);
		}
		do {
			auxlengthold=auxlength-1;
			for (i=auxlength=0; i<=auxlengthold; i++)
				if (aux[i]>tau)
					aux[auxlength++]=aux[i];
				else
					tau+=(tau-aux[i])/(auxlengthold-i+auxlength);
		} while (auxlength<=auxlengthold);
		for (i=0; i<length; i++)
			x[i]=(y[i]-tau>0.0 ? y[i]-tau : (y[i]+tau<0.0 ? y[i]+tau : 0.0));
		if (x==y) free(aux0);
	}
}