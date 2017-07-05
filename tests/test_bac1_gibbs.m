clear
close all
clc

addpath('../')
%%
%addpath('../MCMC_solver2')

M = 10;
N = 15;

y = normrnd(0, 1, [M, 1]);
H = dct(eye(N));
idx = randperm(N);
H = H(idx(1:M),:);

options.sample_X = 1;
options.sample_sigma2 = 0;
options.sample_mu     = 0;

options.niter = 100000;
options.nburn = 10000;
coef = 500;
options.a_mu = 10^(-3);
options.b_mu = 10^(-3);
options.a_sigma2 = 10^(-3);
options.b_sigma2 = 10^(-3);

%%
init.x       = normrnd(0, norm(y) / sqrt(N), [N, 1]);
init.mu      = 5;%400 * (N+1) / (2 * sqrt(N)) * 1 / norm(y);
init.sigma2 = 0.001;%63 / (2 * init.mu * N);

[results, misc] = bac1_gibbs(y - mean(y), H, options, init);
t1 = misc.time;

disp(['c code t=' num2str(t1)]);


[norm(y - mean(y)) norm(y - mean(y) - H * mean(results.x_all, 2))]

figure(1)
clf
hold on
plot(y-mean(y))
plot(H * mean(results.x_all, 2))
