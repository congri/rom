function [E] = isingEnergy(lambdaMat)
%Compute the ising energy of the conductivity field lambda as a feature function

[nr, nc] = size(lambdaMat);

E = 0;
for c = 1:nc
    for r = 1:nr
        if(r > 1)
            if(lambdaMat(r, c) == lambdaMat(r - 1, c))
                E = E + 1;
            else
                E = E - 1;
            end
        else
            %r == 1; periodic boundary conditions
            if(lambdaMat(1, c) == lambdaMat(nr, c))
                E = E + 1;
            else
                E = E - 1;
            end
        end
        if(r < nr)
            if(lambdaMat(r, c) == lambdaMat(r + 1, c))
                E = E + 1;
            else
                E = E - 1;
            end
        else
            %r == nr; periodic boundary conditions
            if(lambdaMat(nr, c) == lambdaMat(1, c))
                E = E + 1;
            else
                E = E - 1;
            end
        end
        if(c > 1)
            if(lambdaMat(r, c) == lambdaMat(r, c - 1))
                E = E + 1;
            else
                E = E - 1;
            end
        else
            %c == 1; periodic boundary conditions
            if(lambdaMat(r, 1) == lambdaMat(r, nc))
                E = E + 1;
            else
                E = E - 1;
            end
        end
        if(c < nc)
            if(lambdaMat(r, c) == lambdaMat(r, c + 1))
                E = E + 1;
            else
                E = E - 1;
            end
        else
            %c == nc; periodic boundary conditions
            if(lambdaMat(r, nc) == lambdaMat(r, 1))
                E = E + 1;
            else
                E = E - 1;
            end
        end
    end
end

end

