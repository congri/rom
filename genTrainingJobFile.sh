NF=256
LENGTHSCALEDIST=delta	#'lognormal' or'delta'
COVARIANCE=squaredExponential
CORRLENGTH1=0.01		#lognormal mu
CORRLENGTH2=0.01		#lognormal sigma
NTRAIN=32
NSTART=rand
VOLFRAC=-1	#Theoretical volume fraction; -1 for uniform random volume fraction
LOCOND=1
HICOND=2
PRIORTYPE=RVM
HYPERPARAM1=[]	#prior hyperparameter
HYPERPARAM2=[]
NCX=\[.25\ .25\ .25\ .25\]
NCY=\[.25\ .25\ .25\ .25\]
BC="[0 800 1200 -2000]"
BC2=\[0\ 800\ 1200\ -2000\]

TESTSAMPLE_LO=1	#for prediction job
TESTSAMPLE_UP=1024

NCORES=8
if [ $NTRAIN -lt $NCORES ]; then
NCORES=$NTRAIN
fi
echo N_cores=
echo $NCORES

NAMEBASE="consecutiveRVM"
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/matlab/projects/rom"
JOBNAME="${NAMEBASE}_randStart_nTrain=${NTRAIN}_Nc=${NCX}_${NCY}"
SPOOL_FILE=/home/constantin/spooledOutput/${DATESTR}_${JOBNAME}
if [ "$LENGTHSCALEDIST" = "lognormal" ]
then
JOBDIR="/home/constantin/matlab/data/fineData/systemSize=${NF}x${NF}/${COVARIANCE}/l=${LENGTHSCALEDIST}_mu=${CORRLENGTH1}sigma=${CORRLENGTH2}_sigmafSq=1/volumeFraction=${VOLFRAC}/locond=${LOCOND}_upcond=${HICOND}/BCcoeffs=${BC2}/${NAMEBASE}/nTrain=${NTRAIN}_Nc=${NCX}_${NCY}_${DATESTR}"
else
JOBDIR="/home/constantin/matlab/data/fineData/systemSize=${NF}x${NF}/${COVARIANCE}/l=${CORRLENGTH1}_sigmafSq=1/volumeFraction=${VOLFRAC}/locond=${LOCOND}_upcond=${HICOND}/BCcoeffs=${BC2}/${NAMEBASE}/nTrain=${NTRAIN}_nStart=${NSTART}_Nc=${NCX}_${NCY}_${DATESTR}"
echo delta length scale
fi

#Create job directory and copy source code
mkdir -p "${JOBDIR}"
cp -r $PROJECTDIR/* "$JOBDIR"
#Remove existing data folder
rm -r $PROJECTDIR/data
#Remove existing predictions file
rm $PROJECTDIR/predictions.mat
#Change directory to job directory; completely independent from project directory
cd "$JOBDIR"
echo $PWD
CWD=$(printf "%q\n" "$(pwd)")
rm job_file.sh

#write job file
printf "#PBS -N $JOBNAME
#PBS -l nodes=1:ppn=$NCORES,walltime=240:00:00
#PBS -o /home/constantin/OEfiles
#PBS -e /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd \"$JOBDIR\"

#Set parameters
sed -i \"7s/.*/        nElFX = $NF;/\" ./ROM_SPDE.m
sed -i \"8s/.*/        nElFY = $NF;/\" ./ROM_SPDE.m
sed -i \"10s/.*/        lowerConductivity = $LOCOND;/\" ./ROM_SPDE.m
sed -i \"11s/.*/        upperConductivity = $HICOND;/\" ./ROM_SPDE.m
sed -i \"13s/.*/        conductivityDistribution = '${COVARIANCE}';/\" ./ROM_SPDE.m
sed -i \"39s/.*/        nStart = $NSTART;             %%first training data sample in file/\" ./ROM_SPDE.m
sed -i \"40s/.*/        nTrain = $NTRAIN;            %%number of samples used for training/\" ./ROM_SPDE.m
sed -i \"62s/.*/        thetaPriorType = '$PRIORTYPE';/\" ./ROM_SPDE.m
sed -i \"63s/.*/        thetaPriorHyperparam = [$HYPERPARAM1 $HYPERPARAM2];/\" ./ROM_SPDE.m
sed -i \"98s/.*/        testSamples = [${TESTSAMPLE_LO}:${TESTSAMPLE_UP}];       %%pick out specific test samples here/\" ./ROM_SPDE.m
sed -i \"140s/.*/        conductivityDistributionParams = {$VOLFRAC [$CORRLENGTH1 $CORRLENGTH2] 1};/\" ./ROM_SPDE.m
sed -i \"147s/.*/        boundaryConditions = '$BC';/\" ./ROM_SPDE.m
sed -i \"152s/.*/        coarseGridVectorX = $NCX;/\" ./ROM_SPDE.m
sed -i \"153s/.*/        coarseGridVectorY = $NCY;/\" ./ROM_SPDE.m


#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r \"trainModel ; quit;\" | tee ${SPOOL_FILE}" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh	#to test in shell

