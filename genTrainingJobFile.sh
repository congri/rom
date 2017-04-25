NF=256
CORRLENGTH=20
NTRAIN=128
NSTART=1	#First training data sample in data file
VOLFRAC=0.2	#Theoretical volume fraction
LOCOND=1
HICOND=10
HYPERPARAM=0	#Lasso sparsity hyperparameter
NC=4
BC="[0 1000 0 0]"
BC2=\[0\ 1000\ 0\ 0\]

DATESTR=`date +%m-%d-%H-%M-%S`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/matlab/projects/rom"
JOBNAME="nTrain=${NTRAIN}_volfrac${VOLFRAC}_lo=${LOCOND}_hi=${HICOND}_Nc=${NC}l=${CORRLENGTH}gamma=${HYPERPARAM}"
JOBDIR="/home/constantin/matlab/data/fineData/systemSize=${NF}x${NF}/correlated_binary/IsoSEcov/l=${CORRLENGTH}_sigmafSq=1/volumeFraction=${VOLFRAC}/locond=${LOCOND}_upcond=${HICOND}/BCcoeffs=${BC2}/localNb_nTrain=${NTRAIN}_Nc=${NC}_${DATESTR}"

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
#PBS -l nodes=1:ppn=16,walltime=240:00:00
#PBS -e /home/constantin/OEfiles
#PBS -o /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to job directory
cd \"$JOBDIR\"
#Set parameters
sed -i \"4s/.*/nStart = $NSTART;/\" ./params/params.m
sed -i \"5s/.*/nTrain = $NTRAIN;/\" ./params/params.m
sed -i \"91s/.*/theta_prior_hyperparamArray = [$HYPERPARAM];/\" ./params/params.m
sed -i \"4s/.*/nf = $NF;/\" ./loadTrainingData.m
sed -i \"12s/.*/bc = '$BC';/\" ./loadTrainingData.m
sed -i \"5s/.*/loCond = $LOCOND;/\" ./loadTrainingData.m
sed -i \"6s/.*/upCond = $HICOND;/\" ./loadTrainingData.m
sed -i \"8s/.*/corrlength = '${CORRLENGTH}';/\" ./loadTrainingData.m
sed -i \"9s/.*/volfrac = '$VOLFRAC';  %%high conducting phase volume fraction/\" ./loadTrainingData.m
sed -i \"2s/.*/nc = $NC;/\" ./params/genCoarseDomain.m


#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r \"trainModel ; quit;\"" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh	#to test in shell

