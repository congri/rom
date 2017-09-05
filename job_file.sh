#PBS -N RVM_nTrain=128_Nc=[.25 .25 .25 .25]_[.25 .25 .25 .25]
#PBS -l nodes=1:ppn=16,walltime=240:00:00
#PBS -e /home/constantin/OEfiles
#PBS -o /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd ""
#Set parameters
sed -i "7s/.*/        nElFX = 256;/" ./ROM_SPDE.m
sed -i "8s/.*/        nElFY = 256;/" ./ROM_SPDE.m
sed -i "10s/.*/        lowerConductivity = 1;/" ./ROM_SPDE.m
sed -i "11s/.*/        upperConductivity = 10;/" ./ROM_SPDE.m
sed -i "35s/.*/        nStart = 1;             %first training data sample in file/" ./ROM_SPDE.m
sed -i "36s/.*/        nTrain = 128;            %number of samples used for training/" ./ROM_SPDE.m
sed -i "58s/.*/        thetaPriorType = 'RVM';/" ./ROM_SPDE.m
sed -i "59s/.*/        thetaPriorHyperparam = [0.4 []];/" ./ROM_SPDE.m
sed -i "129s/.*/        conductivityDistributionParams = {-1 [-3.7 0.3] 1};/" ./ROM_SPDE.m
sed -i "136s/.*/        boundaryConditions = '[0 1000 0 0]';/" ./ROM_SPDE.m
sed -i "140s/.*/        coarseGridVectorX = [.25 .25 .25 .25];/" ./ROM_SPDE.m
sed -i "141s/.*/        coarseGridVectorY = [.25 .25 .25 .25];/" ./ROM_SPDE.m


#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r "trainModel ; quit;"