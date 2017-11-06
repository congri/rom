TESTSAMPLE_LO=1
TESTSAMPLE_UP=1024
CWD=$(printf "%q\n" "$(pwd)")

JOBNAME="prediction"
DATESTR=`date +%m-%d-%H-%M-%S`	#datestring for output name
SPOOL_FILE=/home/constantin/spooledOutput/${DATESTR}_${JOBNAME}

#delete old job file
rm job_file.sh
#write job file
printf "#PBS -N $JOBNAME
#PBS -l nodes=1:ppn=8,walltime=240:00:00
#PBS -o /home/constantin/OEfiles
#PBS -e /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd $CWD
#Set parameters
sed -i \"3s/.*/romObjPred.testSamples = $TESTSAMPLE_LO:$TESTSAMPLE_UP;/\" ./predictionScript.m


#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r \"predictionScript ; quit;\" | tee ${SPOOL_FILE}" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh	#to test in shell
