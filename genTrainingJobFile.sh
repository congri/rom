NF=256
LENGTHSCALEDIST=delta	#'lognormal' or'delta'
COVARIANCE=squaredExponential
CORRLENGTH1=0.01		#lognormal mu
CORRLENGTH2=0.01		#lognormal sigma
NTRAIN=128
NSTART=1	#First training data sample in data file
VOLFRAC=-1	#Theoretical volume fraction; -1 for uniform random volume fraction
LOCOND=1
HICOND=2
PRIORTYPE=RVM
HYPERPARAM1=[]	#prior hyperparameter
HYPERPARAM2=[]
NCX=\[.125\ .125\ .125\ .125\ .125\ .125\ .125\ .125\]
NCY=\[.125\ .125\ .125\ .125\ .125\ .125\ .125\ .125\]
BC="[0 800 1200 -2000]"
BC2=\[0\ 800\ 1200\ -2000\]
NCORES=16
if [ $NTRAIN -lt $NCORES ]; then
$NCORES=$NTRAIN
fi
echo N_cores=
echo $NCORES

NAMEBASE="${COVARIANCE}"
DATESTR=`date +%m-%d-%H-%M-%S`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/matlab/projects/rom"
JOBNAME="${NAMEBASE}_nTrain=${NTRAIN}_Nc=${NCX}_${NCY}"
if [ "$LENGTHSCALEDIST" = "lognormal" ]
then
JOBDIR="/home/constantin/matlab/data/fineData/systemSize=${NF}x${NF}/${COVARIANCE}/l=${LENGTHSCALEDIST}_mu=${CORRLENGTH1}sigma=${CORRLENGTH2}_sigmafSq=1/volumeFraction=${VOLFRAC}/locond=${LOCOND}_upcond=${HICOND}/BCcoeffs=${BC2}/${NAMEBASE}_nTrain=${NTRAIN}_Nc=${NCX}_${NCY}_${DATESTR}"
else
JOBDIR="/home/constantin/matlab/data/fineData/systemSize=${NF}x${NF}/${COVARIANCE}/l=${CORRLENGTH1}_sigmafSq=1/volumeFraction=${VOLFRAC}/locond=${LOCOND}_upcond=${HICOND}/BCcoeffs=${BC2}/${NAMEBASE}_nTrain=${NTRAIN}_Nc=${NCX}_${NCY}_${DATESTR}"
echo delta length scale
fi

#Create job directory and copy source code
mkdir "${JOBDIR}"
cp -r $PROJECTDIR/* "$JOBDIR"
#Remove existing data folder
rm -r $PROJECTDIR/data
#Remove existing predictions file
rm $PROJECTDIR/predictions.mat
#Change directory to job directory; completely independent from project directory
cd "$JOBDIR"
CWD=$(printf "%q\n" "$(pwd)")
rm job_file.sh

#write job file
printf "#PBS -N $JOBNAME
#PBS -l nodes=1:ppn=$NCORES,walltime=240:00:00
#PBS -e /home/constantin/OEfiles
#PBS -o /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd \"$JOBDIR\"
#Set parameters
sed -i \"7s/.*/        nElFX = $NF;/\" ./ROM_SPDE.m
sed -i \"8s/.*/        nElFY = $NF;/\" ./ROM_SPDE.m
sed -i \"10s/.*/        lowerConductivity = $LOCOND;/\" ./ROM_SPDE.m
sed -i \"11s/.*/        upperConductivity = $HICOND;/\" ./ROM_SPDE.m
sed -i \"36s/.*/        nStart = $NSTART;             %%first training data sample in file/\" ./ROM_SPDE.m
sed -i \"37s/.*/        nTrain = $NTRAIN;            %%number of samples used for training/\" ./ROM_SPDE.m
sed -i \"59s/.*/        thetaPriorType = '$PRIORTYPE';/\" ./ROM_SPDE.m
sed -i \"60s/.*/        thetaPriorHyperparam = [$HYPERPARAM1 $HYPERPARAM2];/\" ./ROM_SPDE.m
sed -i \"133s/.*/        conductivityDistributionParams = {$VOLFRAC [$CORRLENGTH1 $CORRLENGTH2] 1};/\" ./ROM_SPDE.m
sed -i \"140s/.*/        boundaryConditions = '$BC';/\" ./ROM_SPDE.m
sed -i \"145s/.*/        coarseGridVectorX = $NCX;/\" ./ROM_SPDE.m
sed -i \"146s/.*/        coarseGridVectorY = $NCY;/\" ./ROM_SPDE.m


#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r \"trainModel ; quit;\"" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh	#to test in shell

