clear
close all
clc

%%

addpath('../utils/random')

M = 1000;
N = 5000;

y = normrnd(0, 1, [M, 1]);
H = eye(N);
idx = randperm(N);
H = H(idx(1:M),:);

options.sample_X = 0;
options.sample_beta = 1;

options.niter = 10000;
options.nburn = 500;
coef = 500;
options.a_mu = 10^(-3);
options.b_mu = 10^(-3);
options.a_beta = 10^(-6);
options.b_beta = 10^(-6);

options.maxEigHH = max(eig(H'*H));

mu_true     = 5 / N;
sigma2_true = 0.1;
x_true      =  demornd(mu_true * N, N, 1);

beta_true = 2 * sigma2_true * N * mu_true;

init.mu      = mu_true;
init.x       = x_true;
init.sigma2  = sigma2_true;
init.beta    = beta_true;

y = H * x_true + normrnd(0, sqrt(sigma2_true), [M, 1]);

%%

[results_pmyula, ~] = bac2_pmyula(y, H, options, init);

figure(1)
clf

histogram(results_pmyula.beta_all, 'EdgeColor', 'none')
line(beta_true * [1 1], ylim)
set(gca, 'box', 'off')
title('pmyula')


set(gcf, 'color', 'w')