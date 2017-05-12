%Generate coarse domain
nX = 5;
nY = 5;
domainc = Domain(nX, nY, [.0625 .0625 .125 .25 .5], [.0625 .0625 .125 .25 .5]);
domainc = setBoundaries(domainc, [2:(2*nX + 2*nY)], Tb, qb);           %ATTENTION: natural nodes have to be set manually
                                                                %and consistently in domainc and domainf
if ~exist('./data/', 'dir')
    mkdir('./data/');
end
filename = './data/domainc.mat';
save(filename, 'domainc');
