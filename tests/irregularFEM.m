%test script for FEM with 'irregular' grid, i.e. with non-uniform squares

nElX = 8;
nElY



D = zeros(2, 2, testDomain.nEl);
%Conductivity matrix D, only consider isotropic materials here
for j = 1:domainc.nEl
    D(:, :, j) =  conductivity(j)*eye(2);
end