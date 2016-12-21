#PBS -N trueMC
#PBS -l nodes=1:ppn=1,walltime=120:00:00
#PBS -o /home/constantin/OEfiles
#PBS -e /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

#Switch to project directory
cd /home/constantin/matlab/projects/cgrom2d

#Run Matlab
/home/constantin/Software/matlab2016b/bin/matlab -nodesktop -nodisplay -nosplash -r "trueMC ; quit;"
