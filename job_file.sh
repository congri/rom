#PBS -N RVM_nTrain=64_Nc=[.25 .25 .25 .25]_[.25 .25 .25 .25]
#PBS -l nodes=1:ppn=16,walltime=240:00:00
#PBS -e /home/constantin/OEfiles
#PBS -o /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd "/home/constantin/matlab/data/fineData/systemSize=256x256/correlated_binary/IsoSEcov/l=delta_mu=0.01sigma=0.01_sigmafSq=1/volumeFraction=-1/locond=1_upcond=100/BCcoeffs=[0 1000 0 0]/RVM_nTrain=64_Nc=[.25 .25 .25 .25]_[.25 .25 .25 .25]_09-12-11-04-15"
#Set parameters
sed -i "7s/.*/        nElFX = 256;/" ./ROM_SPDE.m
sed -i "8s/.*/        nElFY = 256;/" ./ROM_SPDE.m
sed -i "10s/.*/        lowerConductivity = 1;/" ./ROM_SPDE.m
sed -i "11s/.*/        upperConductivity = 100;/" ./ROM_SPDE.m
sed -i "35s/.*/        nStart = 1;             %first training data sample in file/" ./ROM_SPDE.m
sed -i "36s/.*/        nTrain = 64;            %number of samples used for training/" ./ROM_SPDE.m
sed -i "58s/.*/        thetaPriorType = 'RVM';/" ./ROM_SPDE.m
sed -i "59s/.*/        thetaPriorHyperparam = [0 1e-12];/" ./ROM_SPDE.m
sed -i "131s/.*/        conductivityDistributionParams = {-1 [0.01 0.01] 1};/" ./ROM_SPDE.m
sed -i "138s/.*/        boundaryConditions = '[0 1000 0 0]';/" ./ROM_SPDE.m
sed -i "142s/.*/        coarseGridVectorX = [.25 .25 .25 .25];/" ./ROM_SPDE.m
sed -i "143s/.*/        coarseGridVectorY = [.25 .25 .25 .25];/" ./ROM_SPDE.m


#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r "trainModel ; quit;"