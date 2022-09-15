#!/bin/bash
# This bash script traverses all sub-directories and executes rl script for each .cfg file.

ARGS=1
E_WRONGARGS=85

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

rlpath=$(dirname "$SCRIPT_DIR")

if [ $# -ne $ARGS ]
then
    echo "Usage: `basename $0` directory"
    exit $E_WRONGARGS
fi

target_path=$1

echo "Target path = $target_path"

cd $target_path
cfg_files=`find . -name "*.cfg"`
cfg_dirs=`for i in $cfg_files
            do
                dirname $i
            done | uniq`

source $rlpath/rl_env/bin/activate
start_time=`date`
for i in $cfg_dirs
do
    cd $target_path
    cd $i
    echo "Changed path to" `pwd`
    for j in `ls *.cfg`
    do
        python $rlpath/src/rl.py `pwd`/$j >>`pwd`/$j.out 2>&1
    done
    echo "Exiting path" `pwd`
done
wait
deactivate
echo "Start Time = $start_time, End Time = `date`"
