function out = rand_em_gain_new(n_in_array, em_gain)
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

x = zeros(numel(n_in_array), 1);
inds0 = find(~n_in_array);
inds1 = find(n_in_array==1);
inds2 = find(n_in_array==2);
inds3 = find(n_in_array>2);

n0 = length(inds0);
n1 = length(inds1);
n2 = length(inds2);
x(inds0) = rand_em_exact(0, n0, em_gain);
x(inds1) = rand_em_exact(1, n1, em_gain);
x(inds2) = rand_em_exact(2, n2, em_gain);

for i = inds3'
    n_in = n_in_array(i);
    x(i) = rand_em_approx(n_in, em_gain);
end

out = reshape(x, size(n_in_array));
end

function x = rand_em_exact(n_in, numel, g)
% Select a gain distribution based on n_in and generate random numbers.
    rand_array = rand(numel, 1);
    switch n_in
        case 0
            x = zeros(numel, 1);
        case 1
            x = -g * log(1 - rand_array);
        case 2
            x = -g * lambertw(-1, (rand_array-1)/exp(1)) - g;
    end
    x = round(x);
end

function out = rand_em_approx(n_in, g)
% Select a gain distribution based on n_in and generate a single random number.
kmax = 5;
xmin = 0;
xmax = kmax * n_in * g;
nx = 800;
x = linspace(xmin, xmax, nx);

% Basden 2003 probability distribution function is as follows:
% pdf = x.^(n_in-1) .* exp(-x/g) / (g^n_in * factorial(n_in-1))
% Because of the cancellation of very large numbers, first work in log space
logpdf = (n_in-1) * log(x) - x/g - n_in*log(g) - log(factorial(n_in-1));
pdf = exp(logpdf);

% Create a uniformly distributed random number for lookup in CDF
cdf_lookup = rand;

% Map random values to cdf
cdf = cumsum(pdf / sum(pdf));
if cdf_lookup < cdf(1)
    randout = 0;
else
    ihi = find(cdf > cdf_lookup, 1, 'first');
    ilo = ihi - 1;
    xlo = x(ilo);
    xhi = x(ihi);
    clo = cdf(ilo);
    chi = cdf(ihi);
    randout = xlo + (cdf_lookup - clo) * (xhi - xlo)/(chi-clo);
end

out = round(randout);
end
