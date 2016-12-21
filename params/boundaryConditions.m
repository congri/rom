%Generate boundary condition functions

%% Temperature field and gradient generating the boundary conditions
boundaryCoeffs = [-5 3 0 0];
Tb = @(x) boundaryCoeffs(1) + boundaryCoeffs(2)*x(1) + boundaryCoeffs(3)*x(2) + boundaryCoeffs(4)*x(1)*x(2);
qb{1} = @(x) -(boundaryCoeffs(3) + boundaryCoeffs(4)*x);      %lower bound
qb{2} = @(y) (boundaryCoeffs(2) + boundaryCoeffs(4)*y);       %right bound
qb{3} = @(x) (boundaryCoeffs(3) + boundaryCoeffs(4)*x);       %upper bound
qb{4} = @(y) -(boundaryCoeffs(2) + boundaryCoeffs(4)*y);      %left bound