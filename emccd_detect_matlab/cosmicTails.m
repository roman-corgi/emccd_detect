function outMatrix = cosmicTails(frame, pars, props)
% Inputs:
%           frame            : input frame
%           pars             : detector parameters
%           props            : properties from cosmicHits
%
% Output:
%           outmatrix        : input frame + cosmic tails
% 
% S. Miller - 16-Jan-2019

FWCgr   = pars.FWCgr;
matrixh = pars.matrixh;
matrixw = pars.matrixw;

h = props.h;
k = props.k;
r = props.r;

% tails are composed of one rapidly decaying exponential and another
% gradually decaying, with a smooth transition between
n1 = 0.010; % 0.010
n2 = 0.250; % 0.250
n3 = 1.000; % 1.000 controls weight between exponentials
A  = 0.970; % 0.970
B  = 0.030; % 0.030

% create tails
for j=1:length(k)
    rows = max(k(j)-r(j),1):min(k(j)+r(j),matrixh);
    cols = max(h(j)-r(j),1):min(h(j)+r(j),matrixw);
    nonzero = sum(frame(rows,cols)>FWCgr,2)';
    rowsi = rows(nonzero~=0);
    
    for ii = rowsi
        overflowSum = 0;
        colStart = find(frame(ii,cols)>FWCgr, 1, 'first');
        jj = cols(colStart);
        
        while frame(ii,jj) > FWCgr
            overflow = frame(ii,jj) - FWCgr;
            overflowSum = overflowSum + overflow;
            next = min(jj+1, matrixw);
            frame(ii,next) = frame(ii,next) + overflow;
            frame(ii,jj) = FWCgr;
            jj = next;
            tailStart = jj-1;
        end
        scale = 20; % scaling factor to make tails match length of sample images
        taillen = round(overflowSum/FWCgr)*scale;
        tailRowEnd = min(matrixw,tailStart+taillen-1);
        n = linspace(0,1,taillen);
        tailA = A*exp(-n/n1) .* 1./exp(n/n3);
        tailB = B*exp(-n/n2) .* 1./exp(-n/n3);
        tail = frame(ii, tailStart) .* (tailA + tailB);
        frame(ii, tailStart:tailRowEnd) = frame(ii, tailStart:tailRowEnd) + tail(1:(tailRowEnd+1)-tailStart);     
    end
end

outMatrix = frame;
end
