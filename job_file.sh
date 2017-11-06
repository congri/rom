#PBS -N longRun_nTrain=512_Nc=[.125 .125 .125 .125 .125 .125 .125 .125]_[.125 .125 .125 .125 .125 .125 .125 .125]
#PBS -l nodes=1:ppn=8,walltime=240:00:00
#PBS -e /home/constantin/OEfiles
#PBS -o /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd "/home/constantin/matlab/data/fineData/systemSize=256x256/squaredExponential/l=0.01_sigmafSq=1/volumeFraction=-1/locond=1_upcond=2/BCcoeffs=[0 800 1200 -2000]/longRun/nTrain=512_nStart=1_Nc=[.125 .125 .125 .125 .125 .125 .125 .125]_[.125 .125 .125 .125 .125 .125 .125 .125]_10-07-20-31-35"
#Set parameters
sed -i "7s/.*/        nElFX = 256;/" ./ROM_SPDE.m
sed -i "8s/.*/        nElFY = 256;/" ./ROM_SPDE.m
sed -i "10s/.*/        lowerConductivity = 1;/" ./ROM_SPDE.m
sed -i "11s/.*/        upperConductivity = 2;/" ./ROM_SPDE.m
sed -i "13s/.*/        conductivityDistribution = 'squaredExponential';/" ./ROM_SPDE.m
sed -i "39s/.*/        nStart = 1;             %first training data sample in file/" ./ROM_SPDE.m
sed -i "40s/.*/        nTrain = 512;            %number of samples used for training/" ./ROM_SPDE.m
sed -i "62s/.*/        thetaPriorType = 'RVM';/" ./ROM_SPDE.m
sed -i "63s/.*/        thetaPriorHyperparam = [[] []];/" ./ROM_SPDE.m
sed -i "98s/.*/        testSamples = [1:1024];       %pick out specific test samples here/" ./ROM_SPDE.m
sed -i "139s/.*/        conductivityDistributionParams = {-1 [0.01 0.01] 1};/" ./ROM_SPDE.m
sed -i "146s/.*/        boundaryConditions = '[0 800 1200 -2000]';/" ./ROM_SPDE.m
sed -i "151s/.*/        coarseGridVectorX = [.125 .125 .125 .125 .125 .125 .125 .125];/" ./ROM_SPDE.m
sed -i "152s/.*/        coarseGridVectorY = [.125 .125 .125 .125 .125 .125 .125 .125];/" ./ROM_SPDE.m


#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r "trainModel ; quit;"