NF=256
LENGTHSCALEDIST=lognormal	#lognormal or delta
CORRLENGTH1=-3
CORRLENGTH2=0.5
NSET1=1024
NSET2=256
VOLFRAC=-1	#Theoretical volume fraction; negative value leads to uniform random volume fraction
LOCOND=1
UPCOND=10
BC1=0
BC2=1000
BC3=0
BC4=0
#best change boundary conditions in matlab

#Set up file paths
PROJECTDIR="/home/constantin/matlab/projects/rom"
JOBNAME="genDataNf${NF}contrast${LOCOND}-${UPCOND}corrlength=${LENGTHSCALEDIST}${CORRLENGTH1}_${CORRLENGTH2}volfrac${VOLFRAC}"
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
sed -i \"74s/.*/\        conductivityLengthScaleDist = \'${LENGTHSCALEDIST}\';      %%delta for fixed length scale, lognormal for rand/\" ./ROM_SPDE.m
sed -i \"75s/.*/\        conductivityDistributionParams = {$VOLFRAC \[$CORRLENGTH1 $CORRLENGTH2\] 1\};/\" ./ROM_SPDE.m
sed -i \"7s/.*/        nElFX = $NF;/\" ./ROM_SPDE.m
sed -i \"8s/.*/        nElFY = $NF;/\" ./ROM_SPDE.m
sed -i \"27s/.*/        nSets = \[$NSET1 $NSET2\];/\" ./ROM_SPDE.m
sed -i \"4s/.*/ro.lowerConductivity = $LOCOND;/\" ./generateFinescaleData.m
sed -i \"5s/.*/ro.upperConductivity = $UPCOND/\" ./generateFinescaleData.m


#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r \"generateFinescaleData ; quit;\"" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh



