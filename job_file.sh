#PBS -N nTrain=128_volfrac0.10_lo=1_hi=10_Nc=4l=20gamma=30
#PBS -l nodes=1:ppn=16,walltime=240:00:00
#PBS -e /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd "/home/constantin/matlab/data/fineData/systemSize=256x256/correlated_binary/IsoSEcov/l=20_sigmafSq=1/volumeFraction=0.10/locond=1_upcond=10/BCcoeffs=[-50 164 112 -30]/nTrain=128_02-16-10-17-31"
#Set parameters
sed -i "5s/.*/nTrain = 128;/" ./params/params.m
sed -i "62s/.*/theta_prior_hyperparamArray = [30];/" ./params/params.m
sed -i "4s/.*/nf = 256;/" ./loadTrainingData.m
sed -i "12s/.*/bc = '[-50 164 112 -30]';/" ./loadTrainingData.m
sed -i "5s/.*/loCond = 1;/" ./loadTrainingData.m
sed -i "6s/.*/upCond = 10;/" ./loadTrainingData.m
sed -i "8s/.*/corrlength = '20';/" ./loadTrainingData.m
sed -i "9s/.*/volfrac = '0.10';  %high conducting phase volume fraction/" ./loadTrainingData.m
sed -i "2s/.*/nc = 4;/" ./params/genCoarseDomain.m


#Run Matlab
/home/constantin/Software/matlab2016b/bin/matlab -nodesktop -nodisplay -nosplash -r "trainModel ; quit;"