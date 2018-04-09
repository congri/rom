%Test script to check effects of lower conductivity
clear;
restoredefaultpath
addpath('./computation')

ro = ROM_SPDE('genData');
ro = ro.genFineScaleData;

p = ro.plotTrainingInput(1:3, 'input')
ps = ro.plotTrainingOutput(1:3, 'output')