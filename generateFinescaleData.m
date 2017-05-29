%Script to generate finescale data

ro = ROM_SPDE;
ro.conductivityDistributionParams = {.2 [.01 .01] 1};
ro.nElFX = 64;
ro.nElFY = 64;
ro.lowerConductivity = 1;
ro.upperConductivity = 10
ro = ro.genFineScaleData;