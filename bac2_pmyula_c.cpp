#include "mex.h"  /* Always include this */

#include <cmath>
#include <cstring>
#include <random>

std::default_random_engine generator;
std::normal_distribution<double> normal_distribution(0., 1.);


#define datatype double /* type of the elements in y */

double compute_square_norm2(const double *y, const double *H, const double *x, int M, int N);
void compute_grad(const double *y, const double *H, const double *x, double *grad, double sigma2, int M, int N);
void proxCondat(const double *y, double *x, double a, int N);
static void l1ballproj_Condat(datatype* y, datatype* x,
    const unsigned int length, const double a);

void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
        int nrhs, const mxArray *prhs[]) /* Input variables  */
{
    
    if (nrhs != 13)
        mexErrMsgTxt("Wrong number of input arguments.");

    if (nlhs != 2)
        mexErrMsgTxt("Wrong number of output arguments.");    
    
        // inputs
    #define Y_IN            prhs[0]
    #define H_IN            prhs[1]
    #define NIT_IN          prhs[2]
    #define X_IN            prhs[3]
    #define MU_IN           prhs[4]
    #define BETA_IN         prhs[5]    
    #define SAMEPLEX_IN     prhs[6]
    #define SAMEPLEBETA_IN  prhs[7]
    #define A_BETA_IN       prhs[8]
    #define B_BETA_IN       prhs[9]
    #define MAXEIGHH_IN     prhs[10]
    #define THINNING_IN     prhs[11]
    #define SEED_IN         prhs[12]

        // Outputs
    #define CHAIN_X_OUT     plhs[0]
    #define CHAIN_BETA_OUT  plhs[1]
    
    double *y, *H, *x_init, *chain_x, *chain_beta, *buf;// *normrnd_ptr;
    int Nit, thinning;
    bool sample_x, sample_beta;
    int N, M;
    double mu, beta, a_beta, b_beta, maxEigHH;
    
    int i, j, n_it;
    double sigma2;
    double Lf, gamma_myula, lambda_myula;
    double current_error2;
    double seed;
    
    y = mxGetPr(Y_IN);
    H = mxGetPr(H_IN);
    Nit = mxGetScalar(NIT_IN);
    
    x_init = mxGetPr(X_IN);
    mu     = mxGetScalar(MU_IN);
    beta   = mxGetScalar(BETA_IN);
    
    sample_x    = mxGetScalar(SAMEPLEX_IN);
    sample_beta = mxGetScalar(SAMEPLEBETA_IN);
    
    a_beta = mxGetScalar(A_BETA_IN);
    b_beta = mxGetScalar(B_BETA_IN);
    
    maxEigHH = mxGetScalar(MAXEIGHH_IN);
    thinning = mxGetScalar(THINNING_IN);
    seed     = mxGetScalar(SEED_IN);
    generator.seed(seed);
    
    M = mxGetM(H_IN);
    N = mxGetN(H_IN);
    sigma2 = beta / (2. * mu * (double)N);
    double g1_grad[N];
    double x_prox[N];
    double normrnd_ptr[N];
    
    CHAIN_X_OUT    = mxCreateDoubleMatrix(N, Nit, mxREAL); /* Create the output matrix */
    CHAIN_BETA_OUT = mxCreateDoubleMatrix(1, Nit, mxREAL); /* Create the output matrix */
    chain_x    = mxGetPr(CHAIN_X_OUT);
    chain_beta = mxGetPr(CHAIN_BETA_OUT);
    
    double x[N];
    for (j = 0; j < N; j++)
        x[j] = x_init[j];
    
    // Starting loop
    current_error2 = compute_square_norm2(y, H, x, M, N);
    for(n_it = 0; n_it < Nit; n_it++) {

        /** Sample x **/
        if (sample_x) {
            Lf = maxEigHH / sigma2;
            lambda_myula = 1. / Lf;
            gamma_myula  = 1. / (4. * Lf);
            
            double buf_mmse[N];
            for(j = 0; j < N; j++)
                buf_mmse[j] = 0;
            for (int thinning_it = 0; thinning_it < thinning; thinning_it++) {
                
                // Compute gradient
                compute_grad(y, H, x, g1_grad, sigma2, M, N);
                
                // Compute prox
                proxCondat(x, x_prox, lambda_myula * mu * (double)N, N);
                
                // Update
                for(j = 0; j < N; j++) {
                    
                    normrnd_ptr[j] = normal_distribution(generator);
                    
                    x[j] = (1. - gamma_myula / lambda_myula) * x[j]
                            - gamma_myula * g1_grad[j]
                            + gamma_myula / lambda_myula * x_prox[j]
                            + sqrt(2. * gamma_myula) * normrnd_ptr[j];
                    
                    buf_mmse[j] += x[j];
                }
            }
            
            for(j = 0; j < N; j++)
                x[j] = buf_mmse[j] / (double)thinning;
            
            current_error2 = compute_square_norm2(y, H, x, M, N);
        }
        
            // Save x
        for (j=0; j < N; j++)
            chain_x[j + n_it * N] = x[j];
        
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


void proxCondat(const double *y, double *x, double a, int N) {

	double buf[N];

    for (int j = 0; j < N; j++)
        buf[j] = y[j] / a;
    
	l1ballproj_Condat(buf, x, N, 1);
    
    for (int j = 0; j < N; j++)
        x[j] = y[j] - a * x[j];
}

/* Proposed algorithm */
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