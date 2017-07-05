clear
close all
clc

addpath('../')
%%

addpath('../utils/random')

M = 1000;
N = 2000;

y = normrnd(0, 1, [M, 1]);
H = eye(N);
idx = randperm(N);
H = H(idx(1:M),:);

options.sample_X = 0;
options.sample_sigma2 = 1;
options.sample_mu     = 0;

options.niter = 10000;
options.nburn = 500;
coef = 500;
options.a_mu = 10^(-3);
options.b_mu = 10^(-3);
options.a_sigma2 = 10^(-6);
options.b_sigma2 = 10^(-6);

options.maxEigHH = max(eig(H'*H));

mu_true     = 5 / N;
sigma2_true = 0.1;
x_true      =  demornd(mu_true * N, N, 1);

init.mu      = mu_true;
init.x       = x_true;
init.sigma2 = sigma2_true;

y = H * x_true + normrnd(0, sqrt(sigma2_true), [M, 1]);

%%

[results_gibbs, ~]      = bac1_gibbs(y, H, options, init);
[results_pmala, ~]      = bac1_pmala(y, H, options, init);
[results_pmyula, ~]     = bac1_pmyula(y, H, options, init);
[results_gaussian, ~]   = bac1_gaussian(y, H, options, init);
t1 = misc.time;

figure(1)
clf
subplot(2, 2, 1)
histogram(results_gibbs.sigma2_all, 'EdgeColor', 'none')
line(sigma2_true * [1 1], ylim)
set(gca, 'box', 'off')
title('gibbs')

subplot(2, 2, 2)
histogram(results_pmala.sigma2_all, 'EdgeColor', 'none')
line(sigma2_true * [1 1], ylim)
set(gca, 'box', 'off')
title('pmala')

subplot(2, 2, 3)
histogram(results_pmyula.sigma2_all, 'EdgeColor', 'none')
line(sigma2_true * [1 1], ylim)
set(gca, 'box', 'off')
title('pmyula')

subplot(2, 2, 4)
histogram(results_gaussian.sigma2_all, 'EdgeColor', 'none')
line(sigma2_true * [1 1], ylim)
set(gca, 'box', 'off')
title('gaussian')

set(gcf, 'color', 'w')