function out = rand_em_gain(n_in_array, em_gain)
%RAND_EM_GAIN Generate random numbers according to EM gain pdfs.
%   OUT = RAND_EM_GAIN(N_IN_ARRAY, EM_GAIN) returns an array of the same size
%   as n_in_array. Every element in n_in_array is multiplied by
%   em_gain*rand_val, where rand_val is a random number drawn from a specific
%   pdf selected based on the value of the n_in_array element.
%
%   References:
%   [1] http://matlabtricks.com/post-44/generate-random-numbers-with-a-given-distribution
%   [2] https://arxiv.org/pdf/astro-ph/0307305.pdf
%
%   B Nemati and S Miller - UAH - 20-March-2020
if (em_gain<1)
    error('EM gain cannot be set to less than 1');
end

% Find how many values in a array are equal to 0, 1, 2, or >= 3
y = zeros(numel(n_in_array), 1);
inds0 = find(n_in_array==0);
inds1 = find(n_in_array==1);
inds2 = find(n_in_array==2);
inds3 = find(n_in_array>2);

% For n_in of 0, 1, or 2, generate arrays of random numbers according to gain
% equations specific to each n_in
n0 = length(inds0);
n1 = length(inds1);
n2 = length(inds2);
y(inds0) = rand_em_exact(0, n0, em_gain);
y(inds1) = rand_em_exact(1, n1, em_gain);
y(inds2) = rand_em_exact(2, n2, em_gain);

% For n_in of 3 or greater, generate random numbers one by one according to the
% generalized gain equation
for i = inds3'
    n_in = n_in_array(i);
    y(i) = rand_em_approx(n_in, em_gain);
end

out = reshape(y, size(n_in_array));
end

function y = rand_em_exact(n_in, numel, g)
% Select a gain distribution based on n_in and generate random numbers.
    x = rand(numel, 1);
    switch n_in
        case 0
            y = zeros(numel, 1);
        case 1
            y = -g * log(1 - x);
        case 2
            y = -g * lambertw(-1, (x-1)/exp(1)) - g;
    end
    y = round(y);
end

function y = rand_em_approx(n_in, g)
% Select a gain distribution based on n_in and generate a single random number.
kmax = 5;
xmin = eps;
xmax = kmax * n_in * g;
nx = 800;
x = linspace(xmin, xmax, nx);

% Basden 2003 probability distribution function is as follows:
% pdf = x.^(n_in-1) .* exp(-x/g) / (g^n_in * factorial(n_in-1))
% Because of the cancellation of very large numbers, first work in log space
logpdf = (n_in-1)*log(x) - x/g - n_in*log(g) - gammaln(n_in);
pdf = exp(logpdf);
cdf = cumsum(pdf / sum(pdf));

% Create a uniformly distributed random number for lookup in CDF
cdf_lookup = rand*max(cdf) + min(cdf);

% Map random value to cdf
ihi = find(cdf > cdf_lookup, 1, 'first');
ilo = ihi - 1;
xlo = x(ilo);
xhi = x(ihi);
clo = cdf(ilo);
chi = cdf(ihi);
y = xlo + (cdf_lookup - clo) * (xhi - xlo)/(chi-clo);

y = round(y);
end
