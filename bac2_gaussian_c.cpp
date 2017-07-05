#include "mex.h"

#include <cmath>
#include <cstring>
#include <random>

std::default_random_engine generator;
std::normal_distribution<double> normal_distribution(0., 1.);


#define datatype double /* type of the elements in y */


void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
        int nrhs, const mxArray *prhs[]) /* Input variables  */
{
    
    if (nrhs != 11)
        mexErrMsgTxt("Wrong number of input arguments.");

    if (nlhs != 2)
        mexErrMsgTxt("Wrong number of output arguments.");    
    
        // inputs
    #define Y_IN                prhs[0]
    #define H_IN                prhs[1]
    #define NIT_IN              prhs[2]
    #define X_IN                prhs[3]
    #define MU_IN               prhs[4]
    #define BETA_IN             prhs[5]    
    #define SAMEPLEX_IN         prhs[6]
    #define SAMEPLEBETA_IN      prhs[7]
    #define A_BETA_IN           prhs[8]
    #define B_BETA_IN           prhs[9]
    #define SEED_IN             prhs[10]

        // Outputs
    #define CHAIN_X_OUT         plhs[0]
    #define CHAIN_BETA_OUT      plhs[1]
    
    double *y, *H, *x_init, *chain_x, *chain_beta, *buf;
    int Nit;
    bool sample_x, sample_beta;
    int N, M;
    double mu, beta, a_beta, b_beta, sigma2;
    double seed;
    
    int i, j, n_it;
    
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
    
    seed     = mxGetScalar(SEED_IN);
    generator.seed(seed);
    
    M = mxGetM(H_IN);
    N = mxGetN(H_IN);
    sigma2 = beta / (2. * mu * (double)N);

    double g1_grad[N];
    double x_prox[N];
    double normrnd_ptr[N];
    double current_errorVector[M];
    
     /* Create the output matrix */
    if (sample_x)
        CHAIN_X_OUT = mxCreateDoubleMatrix(N, Nit, mxREAL);
    else
        CHAIN_X_OUT = mxCreateDoubleMatrix(N, 1, mxREAL);
    
    if (sample_beta)
        CHAIN_BETA_OUT     = mxCreateDoubleMatrix(1, Nit, mxREAL);
    else
        CHAIN_BETA_OUT     = mxCreateDoubleMatrix(1, 1, mxREAL);
    
    
    chain_x      = mxGetPr(CHAIN_X_OUT);
    chain_beta   = mxGetPr(CHAIN_BETA_OUT);
    
    if (!sample_beta)
        chain_beta[0] = beta;
    
    double x[N];
    for (j = 0; j < N; j++) {
        x[j] = x_init[j];
        
        if (!sample_x)
            chain_x[j] = x_init[j];
    }
    
   
    // norme H
    double normH2[N];
    for(j = 0; j < N; j++) {
        normH2[j] = 0;
        for(i = 0; i < M; i++) {
            double dbuf = H[j * M + i];
            normH2[j] += dbuf * dbuf;
        }
    }
    
    // Compute current error vector
    double current_error2 = 0;
    for (i = 0; i < M; i++) {
        current_errorVector[i] = y[i];
        
        for (j = 0; j < N; j++)
            current_errorVector[i] -= H[i + j * M] * x[j];
        
        current_error2 += current_errorVector[i] * current_errorVector[i];
    }
    
    double sigma_priorX = mu;
    
    /**      Starting loop      **/
    for(n_it = 0; n_it < Nit; n_it++) {

        /** Sample x **/
        if (sample_x) {            
            
            for(j = 0; j < N; j++) {
                                
                // sig = (normHk / sigma_Y)^2 + (1 / sigma_X)^2;
                // sig = sqrt(1 / sig);       
                double var_x = normH2[j] / sigma2 + 1 / (sigma_priorX * sigma_priorX);
                var_x  = 1 / var_x;
                
                // mu  = H(:, n)' * err * (sig / sigma_Y)^2;
                // Update error vector
                for (i = 0; i < M; i++)
                    current_errorVector[i] += H[i + j * M] * x[j];
                
                double mean_x = 0;
                for (i = 0; i < M; i++)
                    mean_x += H[i + j * M] * current_errorVector[i];
                mean_x *= (var_x / sigma2);
                
                x[j] = mean_x +  std::sqrt(var_x) * normal_distribution(generator);
                
                // Update error vector
                current_error2 = 0;
                for (i = 0; i < M; i++) {                    
                    current_errorVector[i] -= H[i + j * M] * x[j];
                    current_error2 += current_errorVector[i] * current_errorVector[i];
                    
                }
            }
        
            // Save x
            for (j=0; j < N; j++)
                chain_x[j + n_it * N] = x[j];
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
