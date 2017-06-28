%Script to generate finescale data; adjust conductivity field parameters in ROM_SPDE

ro = ROM_SPDE;
ro.nElFX = 64;
ro.nElFY = 64;
ro.lowerConductivity = 1;
ro.upperConductivity = 10;
ro = ro.genFineScaleData;