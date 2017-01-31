NF=256
CORRLENGTH=25
NTRAIN=32
VOLFRAC=0.3	#Theoretical volume fraction
LOCOND=1
HICOND=10
NC=8
BC="[-50 164 112 -30]"

DATESTR=`date +%m-%d-%H-%M-%S`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/matlab/projects/rom"
JOBNAME="TrainModel_nTrain=${NTRAIN}_volfrac${VOLFRAC}_locond=${LOCOND}_hicond=${HICOND}_Nc=${NC}"
JOBDIR="/home/constantin/matlab/data/$DATESTR$JOBNAME"

#Create job directory and copy source code
mkdir $JOBDIR
cp -r $PROJECTDIR/* $JOBDIR
#Remove existing data folder
rm -r $PROJECTDIR/data
rm $PROJECTDIR/predictions.mat
#Change directory to job directory; completely independent from project directory
cd $JOBDIR
rm job_file.sh

#write job file
printf "#PBS -N $JOBNAME
#PBS -l nodes=1:ppn=16,walltime=240:00:00
#PBS -o $JOBDIR
#PBS -e $JOBDIR
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd $JOBDIR
#Set parameters
sed -i \"5s/.*/nTrain = $NTRAIN;/\" ./params/params.m
sed -i \"4s/.*/nf = $NF;/\" ./loadTrainingData.m
sed -i \"12s/.*/bc = '$BC';/\" ./loadTrainingData.m
sed -i \"5s/.*/loCond = $LOCOND;/\" ./loadTrainingData.m
sed -i \"6s/.*/upCond = $HICOND;/\" ./loadTrainingData.m
sed -i \"8s/.*/corrlength = '${CORRLENGTH}';/\" ./loadTrainingData.m
sed -i \"9s/.*/volfrac = '$VOLFRAC';  %%high conducting phase volume fraction/\" ./loadTrainingData.m
sed -i \"2s/.*/nc = $NC;/\" ./params/genCoarseDomain.m


#Run Matlab
/home/constantin/Software/matlab2016b/bin/matlab -nodesktop -nodisplay -nosplash -r \"trainModel ; quit;\"" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh	#to test in shell

