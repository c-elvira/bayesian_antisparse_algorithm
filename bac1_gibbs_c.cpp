/* This code need the gsl library
 *
 * How to install gsl and compile this code
 *  OSX     : btrw install gsl -> now located at /usr/local/include/
 *      mex -I/usr/local/include/ -lgsl bac1_gibbs_c.cpp extern/rtnorm.cpp
 *
 *  LINUX   :
 *
 *  WINDOWS :
 *
 */

#include "mex.h"

#include <cmath>
#include <cstring>
#include <random>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_cdf.h>
#include <iostream>

#include "extern/rtnorm.hpp"

std::mt19937 generator;
std::normal_distribution<double>        normal_distribution(0., 1.);
std::uniform_real_distribution<double>  unif_distribution(0.0, 1.0);

//--- GSL random init ---
//gsl_rng_env_setup();                          // Read variable environnement
const gsl_rng_type* type = gsl_rng_default;     // Default algorithm 'twister'
gsl_rng *gsl_gen = gsl_rng_alloc (type);        // Rand generator allocation


#define datatype double /* type of the elements in y */


double compute_square_norm2(const double *y, const double *H, const double *x, int M, int N);

double norminf(const double *x, int N);

double normcdf(double x, double mu, double sigma2);

void uniform_permutation_permutation(int *index, int N);

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
    
    double *y, *H, *x_init, *chain_x, *chain_sigma2, *chain_mu;
    int Nit;
    bool sample_x, sample_mu, sample_sigma2;
    int N, M;
    double mu, sigma2;
    double a_mu, b_mu, a_sigma2, b_sigma2;
    double seed;

    int n_it;
    
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
    seed     = mxGetScalar(SEED_IN);
    generator.seed(seed);

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
    for (int j = 0; j < N; j++) {
        x[j] = x_init[j];
        
        if (!sample_x)
            chain_x[j] = x_init[j];
    }
    
    // norme H
    double normH2[N];
    for(int j = 0; j < N; j++) {
        normH2[j] = 0;
        for(int i = 0; i < M; i++) {
            double dbuf = H[j * M + i];
            normH2[j] += dbuf * dbuf;
        }
    }    
    
    // Compute current error vector
    double current_errorVector[M];
    double current_error2 = 0;
//compute_square_norm2(y, H, x, M, N);
    for (int i = 0; i < M; i++) {
        current_errorVector[i] = y[i];
        
        for (int j = 0; j < N; j++)
            current_errorVector[i] -= H[i + j * M] * x[j];
        
        current_error2 += current_errorVector[i] * current_errorVector[i];
    }
    
    
    
    
    
    
    /**    *****************    **/
    /*       Starting loop       */
    /**    *****************    **/

    
    for(n_it = 0; n_it < Nit; n_it++) {
    
        /** Sample x **/
        if (sample_x) {
            
            int index[N];
            uniform_permutation_permutation(index, N);
            for(int jj = 0; jj < N; jj++) {
                int j = index[jj];
                
                // 1. Update error vector
                for (int i = 0; i < M; i++)
                    current_errorVector[i] += H[i + j * M] * x[j];
                
                // 2. Compute stuff
                x[j] = 0;
                double x_inf_except_j = norminf(x, N);
                
                // 3. Compute truncated gaussian parameters
                double sig2 = sigma2 /  normH2[j];
                double sig  = std::sqrt(sig2);
                    
                    // 3.a mean -> uniform
                double mu_uni = 0;
                for (int i = 0; i < M; i++)
                    mu_uni += H[j * M + i] * current_errorVector[i];
                mu_uni /= normH2[j];
                
                    // 3.b mean -> R_+
                double mu_plus   = 0;
                for (int i = 0; i < M; i++)
                    mu_plus += H[j * M + i] * current_errorVector[i];
                mu_plus = (mu_plus - sigma2 * mu * (double)N) / normH2[j];
                        
                    // 3.c mean -> R_-
                double mu_moins   = 0;
                for (int i = 0; i < M; i++)
                    mu_moins += H[j * M + i] * current_errorVector[i];
                mu_moins = (mu_moins + sigma2 * mu * (double)N) / normH2[j];
                
                
                // 4. Compute probability
                    // 4.a partie uniforme
                double log_u1 = mu_uni * mu_uni / (2 * sig2);
                log_u1 += std::log( 
                        gsl_cdf_ugaussian_P( ( x_inf_except_j - mu_uni) / sig )
                        -
                        gsl_cdf_ugaussian_P( (-x_inf_except_j - mu_uni) / sig )
                        );
                
                    // 4.b partie positive
                double log_u2 = mu_plus * mu_plus / (2 * sig2);        
                log_u2 += mu * (double)N * x_inf_except_j;
                log_u2 += std::log(
                        gsl_cdf_ugaussian_Q( ( x_inf_except_j - mu_plus) / sig )
                        // 1 - gsl_cdf_ugaussian_P( ( x_inf_except_j - mu_plus) / sig )
                        );
        
                    // 4.c partie negative
                double log_u3 = mu_moins * mu_moins / (2 * sig2);        
                log_u3 += mu * (double)N * x_inf_except_j;
                log_u3 += log(
                        gsl_cdf_ugaussian_P( ( -x_inf_except_j - mu_moins) / sig )
                        );
                
                
                // 5. Compute probability
                double omega1 = 1 / (1 
                    + std::exp(log_u2 - log_u1) 
                    + std::exp(log_u3 - log_u1)
                    );
                double omega2 = 1 / (1 
                    + std::exp(log_u1 - log_u2) 
                    + std::exp(log_u3 - log_u2)
                    );
                double omega3 = 1 / (1 
                    + std::exp(log_u1 - log_u3) 
                    + std::exp(log_u2 - log_u3)
                    );
               
                // 6. Select truncated Gaussian
                double u = unif_distribution(generator);
                
                std::pair<double, double> s;
                if (u < omega1) {
                    // 6.a uniform
                    double lima = - x_inf_except_j;
                    double limb = + x_inf_except_j;
                    
                    //std::cout << "uniform " << lima << "  " << limb << std::endl;
                    s = rtnorm(gsl_gen, lima, limb, mu_uni, sig);
                }
                else if (omega1 <= u < omega1 + omega2) {
                    // 6.b R_+
                    double lima = + x_inf_except_j;
                    double limb = + std::numeric_limits<double>::infinity();
                    
                    //std::cout << "positif " << lima << "  " << limb << std::endl;
                    s = rtnorm(gsl_gen, lima, limb, mu_plus, sig);
                }
                else {
                    // 6.b R_-
                    double lima = - std::numeric_limits<double>::infinity();
                    double limb = - x_inf_except_j;
                    
                    //std::cout << "negatif " << lima << "  " << limb << std::endl;
                    s = rtnorm(gsl_gen, lima, limb, mu_moins, sig);
                }
                
                x[j] = s.first;
                
                // 7. Update error vector
                current_error2 = 0;
                for (int i = 0; i < M; i++) {
                    current_errorVector[i] -= H[i + j * M] * x[j];
                    current_error2 += current_errorVector[i] * current_errorVector[i];
                }
            }
            
            // Save x
            for (int j=0; j < N; j++)
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

    for (int i = 0; i < M; i++) {
        buf = y[i];
        
        for (int j = 0; j < N; j++)
            buf -= H[i + j * M] * x[j];
        
        norm += buf * buf;
    }
    return norm;
}


double norminf(const double *x, int N) {
    
    double output = std::abs(x[0]);
    
    for (int j = 0; j < N; j++)
        if (std::abs(x[j]) > output)
            output = std::abs(x[j]);
            
    return output;
}



// Fisher-Yates-Knuth
void uniform_permutation_permutation(int *index, int N) {

    std::uniform_int_distribution<int>  unif_distribution(0, N-1);
    
    for(int j = 0; j < N; j++)
        index[j] = j;
    
    for(int j = 0; j < N; j++) {
        
        int perm = unif_distribution(generator);
        int buf1 = index[j];
        
        index[j] = index[perm];
        index[perm] = buf1;
    }
}