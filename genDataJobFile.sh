NF=256
LENGTHSCALEDIST=delta	#lognormal or delta
COVARIANCE=squaredExponential
CORRLENGTH1=0.01
CORRLENGTH2=0.01
NSET1=4096
NSET2=4096
NSET3=[]
NSET4=[]
NSET5=[]
VOLFRAC=-1	#Theoretical volume fraction; negative value leads to uniform random volume fraction
LOCOND=1
UPCOND=100
BC1=0
BC2=1500
BC3=-500
BC4=1000
CONVECTION=false
#best change boundary conditions in matlab

#Set up file paths
PROJECTDIR="/home/constantin/matlab/projects/rom"
JOBNAME="genDataNf${NF}contrast${LOCOND}-${UPCOND}corrlength=${LENGTHSCALEDIST}${CORRLENGTH1}_${CORRLENGTH2}volfrac${VOLFRAC}_BC=[${BC1}_${BC2}_${BC3}_${BC4}]"
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
#PBS -l nodes=1:ppn=8,walltime=120:00:00
#PBS -o $CWD
#PBS -e $CWD
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd $JOBDIR
#Set parameters
sed -i \"7s/.*/        nElFX = $NF;/\" ./ROM_SPDE.m
sed -i \"8s/.*/        nElFY = $NF;/\" ./ROM_SPDE.m
sed -i \"10s/.*/        lowerConductivity = $LOCOND;/\" ./ROM_SPDE.m
sed -i \"11s/.*/        upperConductivity = $UPCOND;/\" ./ROM_SPDE.m
sed -i \"13s/.*/        conductivityDistribution = \'$COVARIANCE\';/\" ./ROM_SPDE.m
sed -i \"31s/.*/        nSets = \[$NSET1 $NSET2 $NSET3 $NSET4 $NSET5\];/\" ./ROM_SPDE.m
sed -i \"131s/.*/\        useConvection = $CONVECTION;      %%Include a convection term to the pde?/\" ./ROM_SPDE.m
sed -i \"139s/.*/\        conductivityLengthScaleDist = \'${LENGTHSCALEDIST}\';      %%delta for fixed length scale, lognormal for rand/\" ./ROM_SPDE.m
sed -i \"140s/.*/\        conductivityDistributionParams = {$VOLFRAC \[$CORRLENGTH1 $CORRLENGTH2\] 1\};/\" ./ROM_SPDE.m
sed -i \"147s/.*/        boundaryConditions = \'\[$BC1 $BC2 $BC3 $BC4\]\';/\" ./ROM_SPDE.m



#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r \"generateFinescaleData ; quit;\"" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh



