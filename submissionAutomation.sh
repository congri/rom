#!/bin/bash

START=1
GAP=1
END=5

STARTTRAIN=20
GAPTRAIN=4
ENDTRAIN=32

for k in `seq $STARTTRAIN $GAPTRAIN $ENDTRAIN`;
do
	sed -i "6s/.*/NTRAIN=$k/" ./genTrainingJobFile.sh
	for j in `seq $START $GAP $END`;
	do
		#sed -i "7s/.*/NSTART=$j/" ./genTrainingJobFile.sh
		./genTrainingJobFile.sh
	done
done
