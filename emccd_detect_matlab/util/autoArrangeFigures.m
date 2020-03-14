function autoArrangeFigures(NH, NW, monitor_id, rowsFirst)
% INPUT  :
%        NH : number of grid of vertical direction
%        NW : number of grid of horizontal direction
% OUTPUT :
%
% get every figures that are opened now and arrange them.
%
% autoArrangeFigures selects automatically Monitor1.
% If you are dual(or more than that) monitor user, I recommend to set wide
% monitor as Monitor1.
%
% if you want arrange automatically, type 'autoArrangeFigures(0,0)' or 'autoArrangeFigures()'.
%    But maximum number of figures for automatic mode is 27.
%
% if you want specify grid for figures, give numbers for parameters.
%    but if your grid size is smaller than required one for accommodating
%    all figures, this function changes to automatic mode and if more
%    figures are opend than maximum number, then it gives error.
%
% Notes
%  + 2017.1.20 use monitor id(Adam Danz's idea)
%
% leejaejun, Koreatech, Korea Republic, 2014.12.13
% jaejun0201@gmail.com

if nargin < 4
    rowsFirst = 1;
end

if nargin < 3
    monitor_id = 1;
end

if nargin < 2
    NH = 0;
    NW = 0;
end

task_bar_offset = [30 50];

%%
N_FIG = NH * NW;
if N_FIG == 0
    autoArrange = 1;
else
    autoArrange = 0;
end
figHandle = sortFigureHandles(findobj('Type','figure'));
n_fig = size(figHandle,1);
if n_fig <= 0
    warning('figures are not found');
    return
end

screen_sz = get(0,'MonitorPositions');
screen_sz = screen_sz(monitor_id, :);
scn_w = screen_sz(3) - task_bar_offset(1);
scn_h = screen_sz(4) - task_bar_offset(2);
scn_w_begin = screen_sz(1) + task_bar_offset(1);
scn_h_begin = screen_sz(2) + task_bar_offset(2);

if autoArrange==0
    nh = NH;
    nw = NW;
end

if autoArrange == 1
    grid = [2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4;
        3 3 3 3 3 3 3 3 4 4 4 5 5 5 5 5 5 5 5 6 6 6 7 7 7 7 7]';
    
    if n_fig > length(grid)
        warning('too many figures(maximum = %d)',length(grid))
        return
    end
    
    if scn_w > scn_h
        nh = grid(n_fig,1);
        nw = grid(n_fig,2);
    else
        nh = grid(n_fig,2);
        nw = grid(n_fig,1);
    end
end

fig_width = scn_w/nw;
fig_height = scn_h/nh;

for iPic = 1:n_fig
    if rowsFirst
        [row,col] = locateR(iPic,nh,nw);
    else
        [row,col] = locateC(iPic,nh,nw);
    end
    fig_pos = [scn_w_begin + fig_width*(col-1), ...
        scn_h_begin + scn_h-fig_height*row, ...
        fig_width, fig_height];
    set(figHandle(iPic),'OuterPosition',fig_pos);
end
end

function [row,col] = locateR(iPic,totRows,totCols)
row = mod(floor((iPic-1)/totCols),totRows) + 1;
col = mod(iPic-1,totCols) + 1;
end

function [row,col] = locateC(iPic,totRows,totCols)
row = mod(iPic-1,totRows) + 1;
col = mod(floor((iPic-1)/totRows),totCols) + 1;
end

function figSorted = sortFigureHandles(figs)
[tmp, idx] = sort([figs.Number]);
figSorted = figs(idx);
end