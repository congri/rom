%Generate coarse domain
nX = 3;
nY = 3;
domainc = Domain(nX, nY, [.25 .25 .5], [.25 .25 .5]);
domainc = setBoundaries(domainc, [2:(2*nX + 2*nY)], Tb, qb);           %ATTENTION: natural nodes have to be set manually
                                                                %and consistently in domainc and domainf
if ~exist('./data/', 'dir')
    mkdir('./data/');
end
filename = './data/domainc.mat';
save(filename, 'domainc');
