NF=256
LENGTHSCALEDIST=lognormal
CORRLENGTH1=-3
CORRLENGTH2=0.5
NSET1=1024
NSTART=1
NTRAIN=$NSET1
VOLFRAC=-1	#Theoretical volume fraction; negative value leads to uniform random volume fraction
LOCOND=1
UPCOND=10
LATENTDIM=6
#best change boundary conditions in matlab

#Set up file paths
PROJECTDIR="/home/constantin/matlab/projects/rom"
JOBNAME="trainAutoencoder${NF}contrast${LOCOND}-${UPCOND}corrlength=${LENGTHSCALEDIST}${CORRLENGTH1}_${CORRLENGTH2}volfrac${VOLFRAC}latentdim=${LATENTDIM}"
JOBDIR="/home/constantin/matlab/data/$JOBNAME"

#Create job directory and copy source code
rm -r $JOBDIR
mkdir $JOBDIR
#Remove existing data folder
rm -r $PROJECTDIR/data
cp -r $PROJECTDIR/* $JOBDIR
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
sed -i \"15s/.*/ba.latentDim = $LATENTDIM;/\" ./autoencoder/autoEncoderTrainingScript.m
sed -i \"17s/.*/nElFX = $NF;/\" ./autoencoder/autoEncoderTrainingScript.m
sed -i \"18s/.*/nElFY = $NF;/\" ./autoencoder/autoEncoderTrainingScript.m
sed -i \"20s/.*/volFrac = $VOLFRAC;/\" ./autoencoder/autoEncoderTrainingScript.m
sed -i \"22s/.*/lengthscaleDist = \'$LENGTHSCALEDIST\';/\" ./autoencoder/autoEncoderTrainingScript.m
sed -i \"13s/.*/lengthscaleParams = \[$CORRLENGTH1 $CORRLENGTH2\];/\" ./autoencoder/autoEncoderTrainingScript.m
sed -i \"24s/.*/loCond = $LOCOND;/\" ./autoencoder/autoEncoderTrainingScript.m
sed -i \"25s/.*/upCond = $UPCOND;/\" ./autoencoder/autoEncoderTrainingScript.m
sed -i \"27s/.*/nSamples = $NSET1;/\" ./autoencoder/autoEncoderTrainingScript.m
sed -i \"28s/.*/nStart = $NSTART;/\" ./autoencoder/autoEncoderTrainingScript.m
sed -i \"29s/.*/nTrain = $NTRAIN;/\" ./autoencoder/autoEncoderTrainingScript.m

cd autoencoder


#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r \"autoEncoderTrainingScript ; quit;\"" >> job_file.sh

chmod +x job_file.sh
#directly submit job file
qsub job_file.sh
#./job_file.sh



