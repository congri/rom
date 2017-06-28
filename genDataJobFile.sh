NF=256
CORRLENGTH=0.1
NSET1=1024
NSET2=256
VOLFRAC=-1	#Theoretical volume fraction; negative value leads to uniform random volume fraction
LOCOND=1
UPCOND=100
BC1=0
BC2=1000
BC3=0
BC4=0
#best change boundary conditions in matlab

#Set up file paths
PROJECTDIR="/home/constantin/matlab/projects/rom"
JOBNAME="genDataNf${NF}contrast${LOCOND}-${UPCOND}corrlength${CORRLENGTH}volfrac${VOLFRAC}"
JOBDIR="/home/constantin/matlab/data/$JOBNAME"

#Create job directory and copy source code
rm -r $JOBDIR
mkdir $JOBDIR
cp -r $PROJECTDIR/* $JOBDIR
#Remove existing data folder - we generate new data
rm -r $PROJECTDIR/data
#Change directory to job directory; completely independent from project directory
cd $JOBDIR
CWD=$(printf "%q\n" "$(pwd)")
rm job_file.sh

#write job file
printf "#PBS -N $JOBNAME
#PBS -l nodes=1:ppn=16,walltime=120:00:00
#PBS -o $CWD
#PBS -e $CWD
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd $JOBDIR
#Set parameters
sed -i \"74s/.*/\        conductivityDistributionParams = {$VOLFRAC \[$CORRLENGTH $CORRLENGTH\] 1\};/\" ./ROM_SPDE.m
sed -i \"4s/.*/ro.nElFX = $NF;/\" ./generateFinescaleData.m
sed -i \"5s/.*/ro.nElFY = $NF;/\" ./generateFinescaleData.m
sed -i \"6s/.*/ro.lowerConductivity = $LOCOND;/\" ./generateFinescaleData.m
sed -i \"7s/.*/ro.upperConductivity = $UPCOND/\" ./generateFinescaleData.m


#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r \"generateFinescaleData ; quit;\"" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh



