%Generate coarse domain
nX = 4;
nY = 4;
domainc = Domain(nX, nY);
domainc = setBoundaries(domainc, [2:(2*nX + 2*nY)], Tb, qb);           %ATTENTION: natural nodes have to be set manually
                                                                %and consistently in domainc and domainf
if ~exist('./data/', 'dir')
    mkdir('./data/');
end
filename = './data/domainc.mat';
save(filename, 'domainc');
