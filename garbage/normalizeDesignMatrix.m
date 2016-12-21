function [PhiArray] = normalizeDesignMatrix(PhiArray, featureFunctionMean)
%Divides each column in PhiArray by the value stored in featureFunctionMean

PhiArray = PhiArray./featureFunctionMean;

end

