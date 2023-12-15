function save_fig(fig,name)
% save_fig - Function to save a MATLAB figure to a PDF file with correct sizing.
%
% PROTOTYPE
%   save_fig(fig, name)
%
%   fig      - Figure [1x1] to be saved.
%   name     - String [1xN] specifying the desired name of the saved file.
%
% DESCRIPTION:
%   save_fig saves the given figure to a PDF file with correct sizing. It
%   retrieves the figure's aspect ratio, sets appropriate paper size, and
%   automatically adjusts the paper position. The saved file is stored in
%   the 'Report/gfx/' directory with the specified name.
%
% -------------------------------------------------------------------------

WHratio = fig.Position(3)/fig.Position(4); % retrieve current WHratio
widthPos = 15;
heightPos = widthPos/WHratio;

set(fig,'Units','centimeters',...
       'PaperUnits','centimeters',...
       'PaperSize',[widthPos heightPos],...
       'PaperPositionMode','auto',...
       'InvertHardcopy', 'on');
name = strcat('.\Report\gfx\',name);
saveas(fig,name,'pdf')
close(fig)
end


