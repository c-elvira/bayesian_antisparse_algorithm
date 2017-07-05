function [ results, misc ] = bac2_pmyula(y, H, option, init)
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

sample_x = option.sample_X;
sample_beta = option.sample_beta;

x_init    = init.x;
mu_init   = init.mu;
beta_init = init.beta;

a_beta = option.a_beta;
b_beta = option.b_beta;

maxEigHH = option.maxEigHH;

if isfield(option, 'thinning')
    thinning = option.thinning;
else
    thinning = 5;
end

if ~isreal(x_init) || size(x_init, 1) ~= N || size(x_init, 2) ~= 1
    error('init.x should be are a real Nx1 matrix')
end

if ~isreal(mu_init) || ~isreal(beta_init) || numel(mu_init) > 1 || numel(beta_init) > 1
    error('option.mu and option.beta should be real and scalar')
end

if ~isreal(a_beta) || ~isreal(b_beta) || numel(a_beta) > 1 || numel(b_beta) > 1
    error('option.a_beta and option.b_beta should be real and scalar')
end

if ~isreal(maxEigHH) || numel(maxEigHH) > 1
    error('option.maxEigHH should be real and scalar')
end

if ~isreal(thinning) || numel(thinning) > 1
    error('option.thinning should be real and scalar')
end

tic
chain_x = [];
chain_beta = [];
[chain_x, chain_beta] = bac2_pmyula_c(y, H, Nit, ...
    x_init, mu_init, beta_init, ...
    sample_x, sample_beta, ...
    a_beta, b_beta, maxEigHH, thinning, ...
    randi(10^6));
misc.time = toc;
misc.type = 'bac2_pmyula';

results.x_all       = chain_x;
results.beta_all    = chain_beta;
results.mu = init.mu;


