function [Nbubbles] = numberOfObjects(lambdaMat, conductivities, hilo)
%Counts the number of disconnected conductivity bubbles
%   lambdaMat:     2D(!) conductivity image, i.e. 2-dim array

if strcmp(hilo, 'hi')
    cc = bwconncomp((lambdaMat > conductivities(1)));
elseif strcmp(hilo, 'lo')
    cc = bwconncomp((lambdaMat < conductivities(2)));
else
    error('High or low conductivity phase bubbles?')
end

Nbubbles = cc.NumObjects;

end

