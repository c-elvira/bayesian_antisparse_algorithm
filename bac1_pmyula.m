function [ results, misc ] = bac1_pmyula(y, H, option, init)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

misc = struct();
results = struct();

if nargin ~= 4
    error('Wrong number of input arguments.')
end

[M, N] = size(H);

if size(y, 2) > 1
    error('Y should be a M by 1 matrix');
elseif size(y, 1) ~= M
    error('Y should the same number of rows as H');
end

if ~isreal(y) || ~isreal(H)
    error('y and H should be real')
end

Nit = option.nburn + option.niter;

sample_x      = option.sample_X;
sample_mu     = option.sample_mu;
sample_sigma2 = option.sample_sigma2;

x_init      = init.x;
mu_init     = init.mu;
sigma2_init = init.sigma2;

a_mu = option.a_mu;
b_mu = option.b_mu;
a_sigma2 = option.a_sigma2;
b_sigma2 = option.b_sigma2;


maxEigHH = option.maxEigHH;


if isfield(option, 'thinning')
    thinning = option.thinning;
else
    thinning = 5;
end





if ~isreal(x_init) || size(x_init, 1) ~= N || size(x_init, 2) ~= 1
    error('init.x should be are a real Nx1 matrix')
end

if ~isreal(mu_init) || ~isreal(sigma2_init) || numel(mu_init) > 1 || numel(sigma2_init) > 1
    error('option.mu and option.beta should be real and scalar')
end

if ~isreal(a_sigma2) || ~isreal(b_sigma2) || numel(a_sigma2) > 1 || numel(b_sigma2) > 1
    error('option.a_beta and option.b_beta should be real and scalar')
end





tic
chain_x = [];
chain_mu = [];
chain_sigma2 = [];
[chain_x, chain_mu, chain_sigma2] = bac1_pmyula_c(...
    y, H, Nit, ...
    x_init, mu_init, sigma2_init, ...
    sample_x, sample_mu, sample_sigma2, ...
    a_mu, b_mu, a_sigma2, b_sigma2, ...
    thinning, maxEigHH, ...
    randi(10^6));
misc.time = toc;
misc.type = 'bac1_pmyula';

results.x_all      = chain_x;
results.sigma2_all = chain_sigma2;
results.mu_all     = chain_mu;

