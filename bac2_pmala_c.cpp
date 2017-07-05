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
    
    if (nrhs != 14)
        mexErrMsgTxt("Wrong number of input arguments.");

    if (nlhs != 3)
        mexErrMsgTxt("Wrong number of output arguments.");    
    
        // inputs
    #define Y_IN                prhs[0]
    #define H_IN                prhs[1]
    #define NIT_IN              prhs[2]
    #define X_IN                prhs[3]
    #define MU_IN               prhs[4]
    #define BETA_IN             prhs[5]    
    #define SAMPLE_X_IN         prhs[6]
    #define SAMPLE_BETA_IN      prhs[7]
    #define A_BETA_IN           prhs[8]
    #define B_BETA_IN           prhs[9]
    #define THINNING_IN         prhs[10]
    #define WINDOW_IN           prhs[11]
    #define DELTA_IN            prhs[12]
    #define SEED_IN             prhs[13]

        // Outputs
    #define CHAIN_X_OUT         plhs[0]
    #define CHAIN_BETA_OUT      plhs[1]
    #define CHAIN_ARATE_OUT     plhs[2]
    
    double *y, *H, *x_init, *chain_x, *chain_beta, *chain_arate;
    int Nit;
    bool sample_x, sample_beta;
    int N, M;
    double sigma2, mu, beta, a_beta, b_beta;
    int thinning, window;
    double delta;
    double seed;
    
    int i, j, n_it;
    double current_error2;
    
    y = mxGetPr(Y_IN);
    H = mxGetPr(H_IN);
    Nit = mxGetScalar(NIT_IN);
    
    x_init = mxGetPr(X_IN);
    mu     = mxGetScalar(MU_IN);
    beta   = mxGetScalar(BETA_IN);
    
    sample_x      = mxGetScalar(SAMPLE_X_IN);
    sample_beta   = mxGetScalar(SAMPLE_BETA_IN);
    
    a_beta = mxGetScalar(A_BETA_IN);
    b_beta = mxGetScalar(B_BETA_IN);
    seed   = mxGetScalar(SEED_IN);
    generator.seed(seed);
    
    M = mxGetM(H_IN);
    N = mxGetN(H_IN);
    sigma2 = beta / (2. * mu * (double)N);
    
    thinning = mxGetScalar(THINNING_IN);
    window   = mxGetScalar(WINDOW_IN);
    delta   = mxGetScalar(DELTA_IN);

    double g1_grad[N];
    
    double x_star[N];
    double buf_prox[N];
    double x_prox[N];
    double x_star_prox[N];
    
     /* Create the output matrix */
    if (sample_x) {
        CHAIN_X_OUT     = mxCreateDoubleMatrix(N, Nit, mxREAL);
        CHAIN_ARATE_OUT = mxCreateDoubleMatrix(1, Nit, mxREAL);
    }
    else {
        CHAIN_X_OUT     = mxCreateDoubleMatrix(N, 1, mxREAL);
        CHAIN_ARATE_OUT = mxCreateDoubleMatrix(1, 1, mxREAL);
    }
    
    if (sample_beta)
        CHAIN_BETA_OUT = mxCreateDoubleMatrix(1, Nit, mxREAL); /* Create the output matrix */
    else
        CHAIN_BETA_OUT = mxCreateDoubleMatrix(1, 1, mxREAL);

    
    chain_x      = mxGetPr(CHAIN_X_OUT);
    chain_beta   = mxGetPr(CHAIN_BETA_OUT);
    chain_arate  = mxGetPr(CHAIN_ARATE_OUT);
    
    if (!sample_beta)
        chain_beta[0] = beta;    
    
    double x[N];
    for (j = 0; j < N; j++) {
        x[j] = x_init[j];
        
        if (!sample_x) {
            chain_x[j] = x_init[j];
            chain_arate[0] = 1;
        }
    }
    
    current_error2 = compute_square_norm2(y, H, x, M, N);
    
    /**    *****************    **/
    /*       Starting loop       */
    /**    *****************    **/
    unsigned int buf_iter = 0;
    unsigned int buf_accp = 0;
    unsigned int total_iter = 0;
    unsigned int total_accp = 0;
    delta = 0.01;
    for(n_it = 0; n_it < Nit; n_it++) {

        /** Sample x **/
        if (sample_x) {            
            
            // Loop defined by thinning
            for (int th = 0; th < thinning; th++) {
                
                // 1. Compute gradient
                compute_grad(y, H, x, g1_grad, sigma2, M, N);
                
                // 2. Compute prox
                    // 2.a forward
                for (j = 0; j < N; j++)
                    buf_prox[j] = x[j] + delta * g1_grad[j];
                
                    // 2.b backward
                proxCondat(buf_prox, x_prox, 0.5 * delta * mu * (double)N, N);
                
                // 3. Propose
                for (j = 0; j < N; j++)
                    x_star[j] = x_prox[j] + sqrt(delta) * normal_distribution(generator);
                
                // 4. computate acceptation rate
                    // 4.a Compute prox x_star
                        // 4.a.1 forward
                compute_grad(y, H, x_star, g1_grad, sigma2, M, N);
                for (j = 0; j < N; j++)
                    buf_prox[j] = x_star[j] + delta * g1_grad[j];
                
                        // 4.a.2 backward
                proxCondat(buf_prox, x_star_prox, 0.5 * delta * mu * (double)N, N);                
                
                // 4.b Compute alpha
                double log_alpha = 
                        - 0.5 / sigma2 * (compute_square_norm2(y, H, x_star, M, N) - current_error2)
                        - mu * (double)N * ( norminf(x_star, N) - norminf(x, N) )
                        - 0.5 / delta * (distl2(x, x_star_prox, N) - distl2(x_star, x_prox, N));
                
                // 5. accept or reject
                if (std::log(unif_distribution(generator)) < log_alpha) {
                    for (j = 0; j < N; j++)
                        x[j] = x_star[j];
                    
                    current_error2 = compute_square_norm2(y, H, x, M, N);
                    
                    buf_accp++;
                    total_accp++;
                }
                
                // 6. Eventually update rate
                buf_iter++;
                total_iter++;
                
                if (buf_iter == window) {

                    double rate = ((double)buf_accp) / ((double)buf_iter);
                    if (rate > 0.55)
                        delta *= 1.1;
                    else if (rate < 0.45)
                        delta *= 0.9;
                    
                    buf_iter = 0;
                    buf_accp = 0;
                }
                
            }
                    
            // Save x
            
            for (j=0; j < N; j++) {
                chain_x[j + n_it * N] = x[j];
                chain_arate[n_it] = (double)total_accp / (double)total_iter;
            }
            
        }
             
        
        /** Sample beta **/
        if (sample_beta) {
            
            double a1 = a_beta + (double)M / 2;
            double a2 = 1. / (b_beta + mu * (double)N * current_error2);
            std::gamma_distribution<double> gamma_distribution(a1, a2);
            beta = 1. / gamma_distribution(generator);
            
            sigma2 = beta / (2. * mu * (double)N);
        }
        chain_beta[n_it] = beta;
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
    
    for (int j = 1; j < N; j++)
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