%Test script to check effects of lower conductivity

ro = ROM_SPDE('genData');
ro.lowerConductivity = 1;
ro.upperConductivity = 5;
ro.nSets = 8;
ro = ro.genFineScaleData;