N_SAMPLES_P_C=10000
TESTSAMPLE_LO=1
TESTSAMPLE_UP=128
TESTFILEPATH="\/home\/constantin\/matlab\/data\/fineData\/systemSize=256x256\/correlated_binary\/IsoSEcov\/l=10_sigmafSq=1\/volumeFraction=0.1\/locond=1_upcond=10\/BCcoeffs=[-50 164 112 -30]\/set2-samples=128.mat"

JOBNAME="prediction"

#delete old job file
rm job_file.sh
#write job file
printf "#PBS -N $JOBNAME
#PBS -l nodes=1:ppn=16,walltime=240:00:00
#PBS -o $PWD
#PBS -e $PWD
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd $PWD
#Set parameters
sed -i \"2s/.*/nSamples_p_c = $N_SAMPLES_P_C;/\" ./predictionScript.m
sed -i \"3s/.*/testSample_lo = $TESTSAMPLE_LO;/\" ./predictionScript.m
sed -i \"4s/.*/testSample_up = $TESTSAMPLE_UP;/\" ./predictionScript.m
sed -i \"5s/.*/testFilePath = '${TESTFILEPATH}';/\" ./predictionScript.m
sed -i \"15s/.*/save('.\/selfPrediction.mat');/\" ./predictionScript.m


#Run Matlab
/home/constantin/Software/matlab2016b/bin/matlab -nodesktop -nodisplay -nosplash -r \"predictionScript ; quit;\"" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh	#to test in shell
