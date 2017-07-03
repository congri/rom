%%Script for output prediction
romObjPred = ROM_SPDE;
romObjPred = romObjPred.predict;

%save predictions
meanSquaredDistance = romObjPred.meanSquaredDistance;
meanSquaredDistanceError = romObjPred.meanSquaredDistanceError;
meanMahalanobisError = romObjPred.meanMahalanobisError;
save('./predictions.mat', 'meanSquaredDistance', 'meanSquaredDistanceError', 'meanMahalanobisError');
