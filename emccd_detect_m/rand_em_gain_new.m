function out = rand_em_gain_new( NinMtx, EMgain )
% Generate random number according to the EM gain prob density function
% See Basden 2003 paper
%
% Nin = (integer) no. of electrons entering EM register
% EMgain = mean electron multiplication gain of the gain register
% see http://matlabtricks.com/post-44/generate-random-numbers-with-a-given-distribution
%
% this is an approximate, fast algorithm. The mean of the random variates
% created is about 1% low systematically, but for photon counting it should
% be a next-order effect and small
%
% Bijan Nemati - 22-Sep-2018

if (EMgain<1)
    error('EM gain cannot be set to less than 1');
end

[nr, nc] = size(NinMtx);

avgrate = sum(NinMtx(:))/(nr*nc);
if avgrate > 1
    accurateMode = false;
else
    accurateMode = true;
end

x = zeros(nr*nc,1);
outMtx = zeros(nr, nc);
if accurateMode
    ind0 = find(~NinMtx);
    ind1 = find(NinMtx==1);
    ind2 = find(NinMtx==2);
    ind3 = find(NinMtx>2);
    
    n0 = length(ind0);
    n1 = length(ind1);
    n2 = length(ind2);
    n3 = length(ind3);
    x(ind0) = randEM_exact(0, n0, EMgain);
    x(ind1) = randEM_exact(1, n1, EMgain);
    x(ind2) = randEM_exact(2, n2, EMgain);
    
    for i3 = 1 : n3
        Nin = NinMtx(ind3(i3));
        x(ind3(i3)) = randEM_approx(Nin, EMgain);
    end
    outMtx = reshape(x, nr, nc);
else
    for ir = 1:nr
        for ic = 1:nc
            Nin = NinMtx(ir, ic);
            outMtx(ir, ic) = randEM_approx(Nin, EMgain);
        end
    end
end

out = outMtx;
return
%%
function x = randEM_exact(n, Nel, g)
    cvect = rand(Nel, 1);
    switch n
        case 0
            x = zeros(Nel,1);
        case 1
            x = -g * log(1- cvect);
        case 2
            x =  -g * lambertw(-1, (cvect-1)/exp(1)) - g;
 
        otherwise
            error('this function only handles casees of n = 1 or 2!')
    end
    x = round(x);
return
%%
function out = randEM_approx(Nin, EMgain)

if  Nin == 0
    out = 0;
    return;
elseif    Nin < 16
    kmax = 5;
    xmin = eps;
    xmax = kmax * Nin * EMgain;
%     xcorr = 0.5;
    if Nin < 3
        EMgamma = 0;
    else
        EMgamma = gammaln(Nin);
    end
else
    kmax = 4;
    xmin = (Nin - kmax * sqrt(Nin)) * EMgain;
    xmax = (Nin + kmax * sqrt(Nin)) * EMgain;
%     xcorr = 0.3;
    EMgamma = gammaln(Nin);
end

% x = xmin:(xmax-xmin)/99:xmax;
Nx = 800;  % Sam: it looks like setting this to a high number fixes the threshold efficiency issue. 
% it seems to be an issue of speed vs. accuracy. I wonder if we can set a flag to choose cases
x = linspace(xmin, xmax, Nx);
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

% generate random numbers according to pdf 
pdf = pdf / sum(pdf);
cdf = cumsum(pdf);

% create a uniformly distributed random number for lookup in CDF
CDFlookup = rand;

if CDFlookup < cdf(1)
    randout = 0;
else
    ihi = find(cdf > CDFlookup, 1, 'first');
    ilo = ihi - 1;
    xlo = x(ilo); xhi = x(ihi); clo = cdf(ilo); chi = cdf(ihi);
    randout = xlo + (CDFlookup - clo) * ((xhi - xlo)/(chi-clo));
end

out = round(randout) ;
if isempty(out)
    keyboard
end

% % inverse interpolation to achieve P(x) -> x projection of the random values
% if debug
%     figure, plot(x, pdf,'.-', x, cdf, '.:'), title(['Nin = ', num2str(Nin),'     maxNout = ',num2str(xmax)]); %#ok<*UNRCH>
%     legend('pdf', 'cdf');
% end
return
