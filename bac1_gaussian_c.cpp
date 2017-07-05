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
    
    if (nrhs != 13)
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
    #define SEED_IN             prhs[13]

        // Outputs
    #define CHAIN_X_OUT         plhs[0]
    #define CHAIN_MU_OUT        plhs[1]
    #define CHAIN_SIGMA2_OUT    plhs[2]
    
    double *y, *H, *x_init, *chain_x, *chain_sigma2, *chain_mu, *buf;
    int Nit;
    bool sample_x, sample_mu, sample_sigma2;
    int N, M;
    double mu, sigma2, beta, a_mu, b_mu, a_sigma2, b_sigma2;
    double seed;
    
    int i, j, n_it;
    
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
    seed     = mxGetScalar(SEED_IN);
    generator.seed(seed);

    double g1_grad[N];
    double x_prox[N];
    double normrnd_ptr[N];
    double current_errorVector[M];
    
     /* Create the output matrix */
    if (sample_x)
        CHAIN_X_OUT = mxCreateDoubleMatrix(N, Nit, mxREAL);
    else
        CHAIN_X_OUT = mxCreateDoubleMatrix(N, 1, mxREAL);
    
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
        
        
        
        
        
        
        /** Sample mu **/
        if (sample_mu) {
            
            double bufd = 0;
            for (j=0; j < N; j++)
                bufd += x[j] * x[j];
            
            double a1 = a_mu + (double)N / 2;
            double b2 = 1. / (b_mu + 0.5 * bufd);
            
            std::gamma_distribution<double> gamma_distribution(a1, b2);
            
            mu = 1. / gamma_distribution(generator);
        
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
