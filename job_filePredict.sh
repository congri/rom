#PBS -N predict
#PBS -l nodes=node14:ppn=1,walltime=4:00:00
#PBS -e /home/constantin/OEfiles
#PBS -o /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

cd /home/constantin/matlab/data/fineData/systemSize=256x256/correlated_binary/IsoSEcov/l=0.08_sigmafSq=1/volumeFraction=-1/locond=1_upcond=10/BCcoeffs=\[0\ 1000\ 0\ 0\]/RVM_nTrain=128_Nc=\[.125\ .125\ .125\ .125\ .125\ .125\ .125\ .125\]_\[.125\ .125\ .125\ .125\ .125\ .125\ .125\ .125\]_07-03-18-58-52

#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r "predictionScript ; quit;"
