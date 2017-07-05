clear
close all
clc

addpath('../MCMC_solver2')

M = 30;
N = 50;

y = normrnd(0, 1, [M, 1]);
H = dct(eye(N));
idx = randperm(N);
H = H(idx(1:M),:);

options.sample_X = 1;
options.sample_beta = 1;

options.niter = 10000;
options.nburn = 1000;
coef = 500;
options.a_beta = coef * 63 + 2; %3;%10^(-3);
options.b_beta = 63 * (coef * 63 + 1); %2;%10^(-3);
options.maxEigHH = max(eig(H'*H));

init.x       = normrnd(0, norm(y) / sqrt(N), [N, 1]);
init.mu      = 400 * (N+1) / (2 * sqrt(N)) * 1 / norm(y);
init.beta = 63;

[ results, misc ] = antisparse_myula_1(y, H, options, init);
t1 = misc.time;

init.X = init.x;
options.type_sampler_X = 3;
options.get_lambda = @(mu, N) mu*N;

[results, misc] = antisparse2(y, H, options, init);
t2 = misc.time;

disp(['c code t=' num2str(t1)]);
disp(['m code t=' num2str(t2)]);