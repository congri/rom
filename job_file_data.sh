NF=512
CONTRAST=100
JOBNAME="nTrain=$NTRAIN-contrast=$CONTRAST-nf=$NF"
#PBS -N '$JOBNAME'
#PBS -l nodes=1:ppn=16,walltime=120:00:00
#PBS -o /home/constantin/OEfiles
#PBS -e /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

echo $JOBNAME
#Switch to project directory
cd ~/matlab/projects/cgrom2d
#Set parameters
sed -i "35s/.*/    fineData.up = $CONTRAST;   %Change jobfile if you change this line number!/" ./generateFinescaleData.m
sed -i "15s/.*/nf = $NF;       %Should be 2^n/" ./generateFinescaleData.m

#Run Matlab
/home/constantin/Software/matlab2016b/bin/matlab -nodesktop -nodisplay -nosplash -r "generateFinescaleData ; quit;"
