NF=128
CORRLENGTH=5
NSET1=4
NSET2=2
VOLFRAC=0.3	#Theoretical volume fraction
LOCOND=1
HICOND=100

#Set up file paths
PROJECTDIR="/home/constantin/matlab/projects/cgrom2d"
JOBNAME="genDataNf${NF}contrast${LOCOND}-${HICOND}corrlength${CORRLENGTH}volfrac${VOLFRAC}"
JOBDIR="/home/constantin/matlab/data/$JOBNAME"

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
sed -i \"15s/.*/nf = $NF;       %%Should be 2^n/\" ./generateFinescaleData.m
sed -i \"24s/.*/FD = FinescaleData($LOCOND, $HICOND);/\" ./generateFinescaleData.m
sed -i \"26s/.*/FD.nSamples = [$NSET1 $NSET2];/\" ./generateFinescaleData.m
sed -i \"28s/.*/FD.distributionParams = {$VOLFRAC [$CORRLENGTH $CORRLENGTH] 1};/\" ./generateFinescaleData.m

#Run Matlab
/home/constantin/Software/matlab2016b/bin/matlab -nodesktop -nodisplay -nosplash -r \"generateFinescaleData ; quit;\"" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh


