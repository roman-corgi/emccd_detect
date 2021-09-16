function  fighandle = imgshow( M, optionalName)


if ~exist( 'optionalName', 'var' ) || isempty( optionalName )
    figName = inputname(1);
else
    figName = optionalName;
end
if isempty(figName)
    figureName = 'imgShow';
    figureTitle = figName;
else
    validSet = ['A':'Z','a':'z','0':'9'];
    str = figName;
    str(~ismember(str, validSet))= '_';
    figureName = str;
%     figureName = strrep(strrep(strrep(strrep(strrep(strrep(figName,'.','_'),' ','_'), '(','_'), ')','_'), '/','_'), '\','_');
    figureTitle = figName;
end
fighandle = figure('Name',figureName);
 
imagesc(M); axis square; colorbar;
title(figureTitle, 'interpret', 'none')
end

