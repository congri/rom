#PBS -N predict
#PBS -l nodes=1:ppn=1,walltime=4:00:00
#PBS -e /home/constantin/OEfiles
#PBS -o /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r "predictionScript ; quit;"
