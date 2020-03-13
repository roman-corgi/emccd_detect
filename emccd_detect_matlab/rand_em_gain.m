function out = rand_em_gain( Nin, EMgain )
% Generate random number according to the EM gain prob density function
% See Basden 2003 paper

% Nin = (integer) no. of electrons entering EM register
% EMgain = mean electron multiplication gain of the gain register
% see http://matlabtricks.com/post-44/generate-random-numbers-with-a-given-distribution
%
% this is an approximate, fast algorithm. The mean of the random variates
% created is about 1% low systematically, but for photon counting it should
% be a next-order effect and small
%
% Bijan Nemati - 22-Sep-2018

debug = false;

if (EMgain<1)
    error('EM gain cannot be set to less than 1');
end

if Nin < 16
    kmax = 10;
    xmin = eps;
    xmax = kmax * Nin * EMgain;
    xcorr = 0.5;
    if Nin < 3
        EMgamma = 0;
    else
        EMgamma = gammaln(Nin);
    end
else
    kmax = 4;
    xmin = (Nin - kmax * sqrt(Nin)) * EMgain;
    xmax = (Nin + kmax * sqrt(Nin)) * EMgain;
    xcorr = 0.3;
    EMgamma = gammaln(Nin);
end

x = xmin:(xmax-xmin)/99:xmax;

if Nin == 1
    xNin = 0;
else
    xNin = (Nin-1)*log(x);
end

% Basden 2003 probability distribution function
% The prob. dis function is 
%     pdf = x.^(Nin-1) .* exp(-x/EMgain) / (EMgain^Nin * factorial(Nin-1));
% because of the cancellation of very large numbers, first work in log space
logpdf = xNin - x/EMgain - Nin*log(EMgain) - EMgamma;
pdf = exp(logpdf);

% a very ad-hoc correction: compensate for chopped off high tail by skewing the pdf
corrSkew = 1 + xcorr * (0:1/(length(x)-1):1);
pdf = pdf .* corrSkew;

% generate random numbers according to pdf 
pdf = pdf / sum(pdf);
cdf = cumsum(pdf);

% create a uniformly distributed random number for lookup in CDF
CDFlookup = rand(1);

if CDFlookup < cdf(1)
    randout = 0;
else
    ihi = find(cdf > CDFlookup, 1, 'first');
    ilo = ihi - 1;
    xlo = x(ilo); xhi = x(ihi); clo = cdf(ilo); chi = cdf(ihi);
    randout = xlo + (CDFlookup - clo) * ((xhi - xlo)/(chi-clo));
end

out = round(randout) ;

% inverse interpolation to achieve P(x) -> x projection of the random values
if debug
    figure, plot(x, pdf,'.-', x, cdf, '.:'), title(['Nin = ', num2str(Nin),'     maxNout = ',num2str(xmax)]); %#ok<*UNRCH>
    legend('pdf', 'cdf');
end
end

% else
%     % the very large numbers are handled in an approimate way as ~ a gaussian distribution
%     randout = EMgain * max(0, Nin + sqrt(Nin)*randn(1,1));
%
% end
% if isempty(randout)
%     keyboard
% end
% % map randomValues below (cdf(1) to 0)
% ind0 = find(randLookup < cdf(1));
%
% randout = round(interp1(cdf, x, randLookup,'linear'));
% randout(ind0) = 0; %#ok<FNDSB>