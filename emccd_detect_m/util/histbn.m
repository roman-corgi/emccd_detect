function out = histbn(X, varargin)
%Plot a bounded histogram and display underflow and overflow values.
%   HISTBN(X) plots a histogram of X. The default number of bins is
%   sqrt(numel(X)), and the default bounds are set at +/- 5 sigma from the 
%   mean.
%
%   HISTBN(X,NBINS), where NBINS is a positive integer.
%
%   HISTBN(X,NBINS,BOUNDS), where BOUNDS is a two element vector. For
%   default sqrt(numel(X)) number of bins, set NBINS to 'auto'. For no
%   bounds, set BOUNDS to 'all'.
%
%   HISTBN(...,'binWidth',BW,'graphOutliers',GRAPH,'log',LOG), where BW is 
%   a positive number and GRAPH and LOG are booleans. Setting GRAPH to true 
%   will create extra histograms for underflow and/or overflow values. 
%   Setting LOG to true will plot on a logarithmic y scale.
%
%   HISTBN(...,'suppressWarning',SUPPRESS), where SUPPRESS is a boolean.
%   Setting SUPPRESS to true will suppress the warning marker (> !) as well 
%   as the output note about bin bounds being stretched to match bin width.
%
% NOTES
% If binWidth and nbins are both inputs, binWidth will override nbins.
% The objects h, ax, and fig are passed out of the function so the
%   appearance can be customized after calling histbn.
%
% Adapted from histoutline by Matt Foster <ee1mpf@bath.ac.uk>
% B. Nemati and S. Miller - UAH - 10-Jul-2018

%% Parse input
if nargin < 1
    error('Not enough input arguments.');
end
p = inputParser;

defaultNBins = [];
expectedNBins = {'auto'};
checkNBins = @(x) any(x==abs(floor(x)) & isnumeric(x))...
                  || any(validatestring(x,expectedNBins));
defaultBounds = [];
expectedBounds = {'all'};
checkBounds = @(x) (isvector(x) && isnumeric(x) && length(x)==2)...
                   || any(validatestring(x,expectedBounds));
defaultBinWidth = [];
checkBinWidth = @(x) x==abs(x);
defaultGraphOutliers = false;
defaultLog = false;
defaultWarning = false;

addRequired(p,'X',@isnumeric);
addOptional(p,'nbins',defaultNBins,checkNBins);
addOptional(p,'bounds',defaultBounds,checkBounds);
addParameter(p,'binWidth',defaultBinWidth,checkBinWidth);
addParameter(p,'graphOutliers',defaultGraphOutliers,@islogical);
addParameter(p,'log',defaultLog,@islogical);
addParameter(p,'suppressWarning',defaultWarning,@islogical);

parse(p,X,varargin{:});
X = p.Results.X;
nbins = p.Results.nbins;
bounds = p.Results.bounds;
binWidth = p.Results.binWidth;
graphOutliers = p.Results.graphOutliers;
log = p.Results.log;
suppressWarning = p.Results.suppressWarning;

if ~isrow(X)
    X = reshape(X,[1,numel(X)]);
end
if isempty(nbins) || any(strcmp(nbins,'auto'))
    nbins = round(sqrt(length(X)));
end
if any(strcmp(bounds,'all'))
    bounds = [min(X),max(X)];
end

Xlen = length(X);
Xmean = mean(X);
Xstd = std(X);
if isempty(bounds)
    thresh = 5;
    bounds = [Xmean - thresh*Xstd, Xmean + thresh*Xstd];
end

% if bounds are the same, put a space of one between them
if bounds(1) == bounds(2)
    bounds(1) = bounds(1) - 0.5;
    bounds(2) = bounds(2) + 0.5;
end

%% options
font = 'Courier';
textbox = 'fixed';  % options are scaled and fixed

%%
underflowValues = X(X < bounds(1));
overflowValues = X(X > bounds(2));
% cut is the data that will be plotted
cut = X(~ismember(X, [underflowValues, overflowValues]));
cutlen = length(cut);
cutmean = mean(cut);
cutstd = std(cut);
underflow = length(underflowValues);
overflow = length(overflowValues);

overBoundStr = false;
if isempty(binWidth)
    h = histogram(cut, nbins, 'DisplayStyle', 'stairs', 'FaceColor',...
        'none', 'binLimits', bounds);
    binWidth = h.BinWidth;
else
    overBound = mod(bounds(1)-bounds(2),binWidth);
    if ~isempty(bounds) && overBound
        ibound = bounds(2);
        bounds(2) = bounds(2)+overBound;
        if ~suppressWarning
            annotation('textbox', [0.875, 0.05, 0.1, 0.1], 'String', '> !', 'EdgeColor', 'none');
            overBoundStr = sprintf('Right bound stretched from %.3f to %.3f to match bin width.',...
                            ibound, bounds(2));
        end
    end
    h = histogram(cut, 'binWidth', binWidth, 'DisplayStyle', 'stairs',...
        'FaceColor', 'none', 'binLimits', bounds);
end
fig = gcf;
ax = gca;

if log
    ax.YScale = 'log';
    ax.YLim(1) = 0.1;
end

% set x limits to exactly the bounds of the histogram
ax.XLim = bounds;

% set y limits to slightly higher than highest datapoint
adjustYLim(ax, h);

str1 = sprintf('Mean    : % .3f', cutmean);
str2 = sprintf('Std Dev : % .3f', cutstd);
str3 = sprintf('Entries : % d', cutlen);
str4 = sprintf('Underflow : % d', underflow);
str5 = sprintf('Overflow  : % d', overflow);
str = {str1, str2, str3, str4, str5};

switch textbox
    case 'scaled'
        x_txt = 0.61;
        y_txt = 0.8;
        fx_txt = 0.1;
        fy_txt = 0.1;

        font_size = 0.03;
        dim_txt = [x_txt, y_txt, fx_txt, fy_txt];

        annotation('textbox', 'String',str, 'FontUnits','normalized',...
            'FontSize',font_size, 'FontName',font, 'Position',dim_txt,...
            'Units','normalized', 'LineWidth',0.5);

    case 'fixed'
        l(1) = line(nan, nan, 'Color','none');
        l(2) = line(nan, nan, 'Color','none');
        l(3) = line(nan, nan, 'Color','none');
        l(4) = line(nan, nan, 'Color','none');
        l(5) = line(nan, nan, 'Color','none');

        [hlegend, icons, ~, ~] = legend(l, str, 'FontSize',10, 'FontName',font);
        set(hlegend.BoxFace, 'ColorType','truecoloralpha', 'ColorData',...
            uint8(255*[0.8; 0.8; 0.8; 0.2]));
        p1 = icons(1).Position;
        p2 = icons(2).Position;
        p3 = icons(3).Position;
        p4 = icons(4).Position;
        p5 = icons(5).Position;

        left = 0.15;
        icons(1).Position = [left, p1(2), 0];
        icons(2).Position = [left, p2(2), 0];
        icons(3).Position = [left, p3(2), 0];
        icons(4).Position = [left, p4(2), 0];
        icons(5).Position = [left, p5(2), 0];
end

ylabel(sprintf('Entries / Bin  (size %.3f)', binWidth));

if graphOutliers
    if underflow ~= 0
        figure;
        hunder = histogram(underflowValues, 'DisplayStyle', 'stairs',...
                 'FaceColor', 'none','binWidth',binWidth); 
        title('Underflow');
        ylabel(sprintf('Entries / Bin  (size %.3f)', binWidth));
        figunder = gcf;
        axunder = gca;
        if log
            axunder.YScale = 'log';
            axunder.YLim(1) = 0.1;
        end
        axunder.XLim(2) = hunder.BinLimits(2);
        
        adjustYLim(axunder, hunder);
        
        set(axunder, 'FontName',font);
        
        out.hunder = hunder;
        out.figunder = figunder;
        out.axunder = axunder;
    end
    if overflow ~= 0
        figure;
        hover = histogram(overflowValues, 'DisplayStyle', 'stairs',...
                'FaceColor', 'none','binWidth',binWidth); 
        title('Overflow');
        ylabel(sprintf('Entries / Bin  (size %.3f)', binWidth));
        figover = gcf;
        axover = gca;
        if log
            axover.YScale = 'log';
            axover.YLim(1) = 0.1;
        end
        axover.XLim(1) = hover.BinLimits(1);
        
        adjustYLim(axover, hover);
        
        set(axover, 'FontName',font);
        
        out.hover = hover;
        out.figover = figover;
        out.axover = axover;
    end
end
figure(fig);

set(ax, 'FontName',font);

out.Data = h.Data;
out.Values = h.Values;
out.NumBins = h.NumBins;
out.BinEdges = h.BinEdges;
out.BinWidth = h.BinWidth;
out.BinLimits = h.BinLimits;
out.handle = h;
out.fig = fig;
out.ax = ax;

out.mean = cutmean;
out.sdev = cutstd;
out.total = Xlen;
out.dataCut = cutlen;
out.underflow = underflow;
out.overflow = overflow;
out.underflowValues = underflowValues;
out.overflowValues = overflowValues;
if overBoundStr
    out.note = overBoundStr;
end
end

function adjustYLim(ax, h)
%Adjust y-axis upper bound to be slightly higher than the highest datapoint.
tickWidth = ax.YTick(end)-ax.YTick(end-1);
if ax.YLim(2)-tickWidth <= max(h.Values)
    ax.YLim(2) = ax.YLim(2)+tickWidth;
end

end