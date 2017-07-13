function [lambdak] = getCoarseElementConductivity(domainc, domainf, conductivity)
E = zeros(domainf.nEl, 1);
e = 1;  %element number
for row_fine = 1:domainf.nElY
    %coordinate of lower boundary of fine element
    y_coord = domainf.cum_lElY(row_fine);
    row_coarse = sum(y_coord >= domainc.cum_lElY);
    for col_fine = 1:domainf.nElX
        %coordinate of left boundary of fine element
        x_coord = domainf.cum_lElX(col_fine);
        col_coarse = sum(x_coord >= domainc.cum_lElX);
        E(e) = (row_coarse - 1)*domainc.nElX + col_coarse;
        e = e + 1;
    end
end

E = reshape(E, domainf.nElX, domainf.nElY);
pltFineToCoarse = false;
if pltFineToCoarse
    figure
    imagesc(Phi.EMat)
    pause
end

if iscell(conductivity)
    nTrain = numel(conductivity);
else
    nTrain = size(conductivity, 2);
end

lambdak = cell(nTrain, domainc.nEl);

for s = 1:nTrain
    %inputs belonging to same coarse element are in the same column of xk. They are ordered in
    %x-direction.
    %Get conductivity fields in coarse cell windows
    if iscell(conductivity)
        conductivityMat = reshape(conductivity{s}, sqrt(domainf.nEl), sqrt(domainf.nEl));
    else
        conductivityMat = reshape(conductivity(:, s), sqrt(domainf.nEl), sqrt(domainf.nEl));
    end
    for i = 1:domainc.nEl
        indexMat = (E == i);
        lambdakTemp = conductivityMat.*indexMat;
        %Cut elements from matrix that do not belong to coarse cell
        lambdakTemp(~any(lambdakTemp, 2), :) = [];
        lambdakTemp(:, ~any(lambdakTemp, 1)) = [];
        lambdak{s, i} = lambdakTemp;
    end
end
end