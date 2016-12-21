NTRAIN=16
CONTRAST=100
GAMMAB=.0005


DATESTR=`date +%m%d%H%M%S`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/matlab/projects/cgrom2d"
JOBNAME="priorTest$GAMMAB"
JOBDIR="/home/constantin/matlab/data/$DATESTR$JOBNAME"

#Create job directory and copy source code
mkdir $JOBDIR
cp -r $PROJECTDIR/* $JOBDIR
#Remove existing data folder - we generate new data
rm -r $PROJECTDIR/data
#Change directory to job directory; completely independent from project directory
cd $JOBDIR
rm job_file.sh

#write job file
printf "#PBS -N $JOBNAME
#PBS -l nodes=1:ppn=16,walltime=120:00:00
#PBS -o $JOBDIR
#PBS -e $JOBDIR
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd $JOBDIR
#Set parameters
sed -i \"4s/.*/nTrain = $NTRAIN;/\" ./params/params.m
sed -i \"38s/.*/prior_hyperparam = [0; $GAMMAB];/\" ./params/params.m
sed -i \"6s/.*/jobname = '$JOBNAME';/\" ./params/params.m
sed -i \"5s/.*/contrast = $CONTRAST;/\" ./loadTrainingData.m


#Run Matlab
/home/constantin/Software/matlab2016b/bin/matlab -nodesktop -nodisplay -nosplash -r \"trainModel ; quit;\"" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh

