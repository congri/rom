function [out] = meanImageProps(lambdaMat, conductivities, hilo, property, meanVarMaxMin)
%Uses built-in Matlab 'regionprops' to return average property values of binary image bubbles
%   lambda:     2-dim conductivity image
%   conductivities: loCond in first, upCOnd in second entry
%   hilo:       property for high or low phase bubbles?
%   property:
%       'Area':     Mean area of bubble


%Convert lambda to binary image
if strcmp(hilo, 'hi')
    lambdaMat = (lambdaMat > conductivities(1));
elseif strcmp(hilo, 'lo')
    lambdaMat = (lambdaMat < conductivities(2));
else
    error('Property of high or low conducting phase?')
end

props = regionprops(lambdaMat, property);
props = struct2cell(props);
m = 0;
mSq = 0;
Max = -Inf;
Min = Inf;
for i = 1:size(props, 2)
    m = m + props{i};
    mSq = mSq + props{i}^2;
    if(props{i} > Max)
        Max = props{i};
    end
    if(props{i} < Min)
        Min = props{i};
    end
end
if(size(props, 2) == 0)
    m = 0;
    mSq = 0;
    Max = 0;
    Min = 0;
else
    m = m/size(props, 2);
    mSq = mSq/size(props, 2);
end
v = mSq - m^2;

if strcmp(meanVarMaxMin, 'mean')
    out = m;
elseif strcmp(meanVarMaxMin, 'var')
    out = v;
elseif strcmp(meanVarMaxMin, 'max')
    out = Max;
elseif strcmp(meanVarMaxMin, 'min')
    out = Min;
else
    error('Mean or variance of lambda bubble property?')
end


end

