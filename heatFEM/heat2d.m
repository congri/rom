function [Out] = heat2d(domain, D, convectionField)
%2D heat conduction main function
%Gives back temperature on point x

%get_loc_stiff as nested function for performance
Dmat = spalloc(8, 8, 16);
if(nargin > 2)
    convectionFieldMat = spalloc(8, 4, 8);
end
    function [diffusionStiffness, convectionStiffness] = get_loc_stiff2(Bvec, D, convectionMatrix, cField)
        %Gives the local stiffness matrix
        
        Dmat(1:2, 1:2) = D;
        Dmat(3:4, 3:4) = D;
        Dmat(5:6, 5:6) = D;
        Dmat(7:8, 7:8) = D;
        
        diffusionStiffness = Bvec'*Dmat*Bvec;
        if(nargin > 2)
            convectionFieldMat(1:2, 1) = cField;
            convectionFieldMat(3:4, 2) = cField;
            convectionFieldMat(5:6, 3) = cField;
            convectionFieldMat(7:8, 4) = cField;
            convectionStiffness = convectionMatrix*convectionFieldMat;
        end
    end


%Compute local stiffness matrices, once and for all
Out.diffusionStiffness = zeros(4, 4, domain.nEl);
if(nargin > 2)
    Out.convectionStiffness = Out.diffusionStiffness;
end
for e = 1:domain.nEl
    if(nargin > 2)
        [Out.diffusionStiffness(:, :, e), Out.convectionStiffness(:, :, e)] =...
            get_loc_stiff2(domain.Bvec(:, :, e), D(:, :, e), domain.convectionMatrix(:, :, e), convectionField(:, e));
    else
        Out.diffusionStiffness(:, :, e) = get_loc_stiff2(domain.Bvec(:, :, e), D(:, :, e));
    end
end

%Global stiffness matrix
localStiffness = Out.diffusionStiffness;
if(nargin > 2)
    localStiffness = localStiffness + Out.convectionStiffness;
end
Out.globalStiffness = get_glob_stiff2(domain, localStiffness);
%Global force vector
Out.globalForce = get_glob_force(domain, localStiffness);

%Finally solving the equation system
Out.naturalTemperatures = Out.globalStiffness\Out.globalForce;


%Temperature field
Tf = zeros(domain.nNodes, 1);
Tf(domain.id) = Out.naturalTemperatures;
Tff = zeros(domain.nElX + 1, domain.nElY + 1);

for i = 1:domain.nNodes
    Tff(i) = Tf(i);
    if(any(i == domain.essentialNodes))
        %node i is essential
        Tff(i) = domain.essentialTemperatures(i);
    end
end

Tff = Tff';
Out.Tff = Tff;


end