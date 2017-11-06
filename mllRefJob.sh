#PBS -N mllRef
#PBS -l nodes=node15:ppn=1,walltime=240:00:00
#PBS -o /home/constantin/OEfiles
#PBS -e /home/constantin/OEfiles
#PBS -m abe
#PBS -M mailscluster@gmail.com

N=4

NAMEBASE="mllReference_N=${N}"
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/matlab/projects/rom"
JOBNAME="${NAMEBASE}_${DATESTR}"
SPOOL_FILE=/home/constantin/spooledOutput/${JOBNAME}

JOBDIR="/home/constantin/matlab/data/mllRef/${JOBNAME}"

#Remove existing data folder
rm -r $PROJECTDIR/data

#Remove existing predictions file
rm $PROJECTDIR/predictions.mat

#Create job directory and copy source code
mkdir -p "${JOBDIR}"
cp -r $PROJECTDIR/* "$JOBDIR"

#Change directory to job directory; completely independent from project directory
cd "$JOBDIR"
echo $PWD
CWD=$(printf "%q\n" "$(pwd)")
rm job_file.sh

#set N
sed -i "3s/.*/N = ${N};/" ./mllRefScript.m


#cd /home/constantin/matlab/projects/rom
#Run Matlab
/home/matlab/R2017a/bin/matlab -nodesktop -nodisplay -nosplash -r "mllRefScript ; quit;" | tee ${SPOOL_FILE}
