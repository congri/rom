%%Script for output prediction
romObjPred = ROM_SPDE('');
romObjPred.testSamples = 1:1024;
romObjPred = romObjPred.predict;

predMetrics.meanSqDist = romObjPred.meanSquaredDistance;
predMetrics.meanLogLikelihood = romObjPred.meanLogLikelihood;
predMetrics.meanPerp = romObjPred.meanPerplexity;
predMetrics.maha = romObjPred.meanMahalanobisError;
predMetrics.meanSqDistField = romObjPred.meanSquaredDistanceField;

%save predictions
save('./predictions.mat', 'predMetrics');
