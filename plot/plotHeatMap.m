function out = plotHeatMap(X, Y, Tff, domain, physical)
f = figure('name','2D heat map');
% set(f, 'Position', [10, 350, 720, 540]);
contourf(X,Y,Tff,256,'LineColor','none');
% h = pcolor(X,Y,Tff);
axis square
% axis off
% set(h, 'edgecolor', 'none')
title('2D heat map');
xlabel('x');
ylabel('y');
% cb = colorbar;
% ylabel(cb,'T');
set(gca,'FontSize',14) 
% colormap('jet')
dim = [.25 .6 .4 .3];
%concatenate textbox string
if(strcmp(domain.boundaries(1),'essential'))
    s = 'T_{lo} =';
    s = [s ' ' num2str(physical.Tb(1))];
else
    s = 'q_{lo} = ';
    s = [s ' ' num2str(physical.qb(1))];
end
if(strcmp(domain.boundaries(2),'essential'))
    s = [s ' ' '\nT_{r} ='];
    s = [s ' ' num2str(physical.Tb(2))];
else
    s = [s ' ' '\nq_{r} = '];
    s = [s ' ' num2str(physical.qb(2))];
end
if(strcmp(domain.boundaries(3),'essential'))
    s = [s ' ' '\nT_{u} ='];
    s = [s ' ' num2str(physical.Tb(3))];
else
    s = [s ' ' '\nq_{u} = '];
    s = [s ' ' num2str(physical.qb(3))];
end
if(strcmp(domain.boundaries(4),'essential'))
    s = [s ' ' '\nT_{le} ='];
    s = [s ' ' num2str(physical.Tb(4))];
else
    s = [s ' ' '\nq_{le} = '];
    s = [s ' ' num2str(physical.qb(4))];
end
str = sprintf(s);
annotation('textbox',dim,'String',str,'FitBoxToText','on');

out = f;