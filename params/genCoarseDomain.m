%Generate coarse domain
nc = 2;
domainc = Domain(nc, nc);
domainc = setBoundaries(domainc, [2:(4*nc)], Tb, qb);           %ATTENTION: natural nodes have to be set manually
                                                                %and consistently in domainc and domainf
if ~exist('./data/', 'dir')
    mkdir('./data/');
end
filename = './data/domainc.mat';
save(filename, 'domainc');