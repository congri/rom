function [L] = triuTransform(v)
%Transforms vector v in upper triangular matrix L according to
% L = (v_1 v_2 v_4)
%     ( 0  v_3 v_5)
%     ( 0   0  v_6)
%and accordingly for higher dimensions


dim_v = numel(v);
dim_L = -.5 + .5*sqrt(1 + 8*dim_v);
if mod(dim_L, 1)
    error('wrong number of elements in v to transform it to square triangular matrix')
end

L = zeros(dim_L);
row = 1;
col = 1;
for i = 1:dim_v
    L(row, col) = v(i);
    if row < col
        row = row + 1;
    else
        row = 1;
        col = col + 1;
    end
end

end

