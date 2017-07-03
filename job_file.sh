#PBS -N RVM_nTrain=128_Nc=[.25 .25 .25 .25]_[.25 .25 .25 .25]
#PBS -l nodes=1:ppn=16,walltime=240:00:00
#PBS -e /home/constantin/OEfiles
#PBS -o /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd "/home/constantin/matlab/data/fineData/systemSize=256x256/correlated_binary/IsoSEcov/l=0.08_sigmafSq=1/volumeFraction=.2/locond=1_upcond=10/BCcoeffs=[0 1000 0 0]/RVM_nTrain=128_Nc=[.25 .25 .25 .25]_[.25 .25 .25 .25]_06-28-19-22-56"
#Set parameters
sed -i "33s/.*/        nStart = 1;             %first training data sample in file/" ./ROM_SPDE.m
sed -i "34s/.*/        nTrain = 128;            %number of samples used for training/" ./ROM_SPDE.m
sed -i "70s/.*/theta_prior_hyperparamArray = [0.4];/" ./params/params.m
sed -i "7s/.*/        nElFX = 256;/" ./ROM_SPDE.m
sed -i "8s/.*/        nElFY = 256;/" ./ROM_SPDE.m
sed -i "77s/.*/        boundaryConditions = '[0 1000 0 0]';/" ./ROM_SPDE.m
sed -i "10s/.*/        lowerConductivity = 1;/" ./ROM_SPDE.m
sed -i "11s/.*/        upperConductivity = 10;/" ./ROM_SPDE.m
sed -i "74s/.*/        conductivityDistributionParams = {.2 [0.08 0.08] 1};      %for correlated_binary:/" ./ROM_SPDE.m
sed -i "81s/.*/        coarseGridVectorX = [.25 .25 .25 .25];/" ./ROM_SPDE.m
sed -i "82s/.*/        coarseGridVectorY = [.25 .25 .25 .25];/" ./ROM_SPDE.m


#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r "trainModel ; quit;"