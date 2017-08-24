%Script to generate finescale data; adjust conductivity field parameters in ROM_SPDE

ro = ROM_SPDE('genData');
ro.lowerConductivity = 1;
ro.upperConductivity = 1;
ro = ro.genFineScaleData;
