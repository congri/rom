function [Equations, LocalNode] = get_equations(nElements, lm)
%Equation number array for sparse global stiffness assembly

localNodeInit = 1:4;
%preallocate
Equations = zeros(16*nElements, 2);
LocalNode = zeros(16*nElements, 3);
nEq = 0;
for e = 1:nElements
    equationslm = lm(e,localNodeInit);
    equations = equationslm(equationslm > 0);
    localNode = localNodeInit(equationslm > 0);
    prevnEq = nEq;
    nEq = nEq + numel(equations)^2;

    [Equations1, Equations2] = meshgrid(equations);
    Equations((prevnEq + 1):nEq, :) = [Equations1(:) Equations2(:)];

    [LocalNode1, LocalNode2] = meshgrid(localNode);
    LocalNode((prevnEq + 1):nEq, :) = [LocalNode1(:) LocalNode2(:) repmat(e, length(equations)^2, 1)];
end

%Shrink to fit
Equations((nEq + 1):end, :) = [];
LocalNode((nEq + 1):end, :) = [];

end

