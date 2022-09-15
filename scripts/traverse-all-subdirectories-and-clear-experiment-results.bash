#!/bin/bash
# This bash script traverses all sub-directories and removes experiment results.

ARGS=1
E_WRONGARGS=85

if [ $# -ne $ARGS ]
then
    echo "Usage: `basename $0` directory"
    exit $E_WRONGARGS   
fi

target_path=$1
echo "all test results in the path (and its subpaths) will be deleted for:"
echo "$target_path"

read -p "Are you sure (y/n)? " -n 1 -r
echo 
if [[ $REPLY =~ ^[Yy]$ ]]
then
    cd $target_path
    find $target_path -iname "*.rlout" -exec rm -f '{}' ';'
    find $target_path -iname "*.rlest" -exec rm -f '{}' ';'
    find $target_path -iname "*.rlasi" -exec rm -f '{}' ';'
    find $target_path -iname "*.rlsvf" -exec rm -f '{}' ';'
    find $target_path -iname "*.rlopts" -exec rm -f '{}' ';'
    find $target_path -iname "*.rldd" -exec rm -f '{}' ';'
    find $target_path -iname "*.rlnsm" -exec rm -f '{}' ';'
    find $target_path -iname "*.rlgraph" -exec rm -f '{}' ';'
    find $target_path -iname "*.rltree" -exec rm -f '{}' ';'
    find $target_path -iname "*.rlobsgraph" -exec rm -f '{}' ';'
    find $target_path -iname "*.odata" -exec rm -f '{}' ';'
    find $target_path -iname "*.rlobsstats" -exec rm -f '{}' ';'
    find $target_path -iname "*.rloutSingle" -exec rm -f '{}' ';'
    find $target_path -iname "*.rlpol" -exec rm -f '{}' ';'
    find $target_path -iname "*.rlhist" -exec rm -f '{}' ';'
    find $target_path -iname "*.rlvisit" -exec rm -f '{}' ';'
    find $target_path -iname "*.rsinfo" -exec rm -f '{}' ';'
    find $target_path -iname "*.ps" -exec rm -f '{}' ';'
    find $target_path -iname "*.eps" -exec rm -f '{}' ';'
    find $target_path -iname "*.gv" -exec rm -f '{}' ';'
    find $target_path -iname "*.plt" -exec rm -f '{}' ';'
    find $target_path -iname "*.pdf" -exec rm -f '{}' ';'
    find $target_path -iname "*.png" -exec rm -f '{}' ';'
    find $target_path -iname "*.txt" -exec rm -f '{}' ';'
    find $target_path -iname "*.rlshist" -exec rm -f '{}' ';'
    find $target_path -iname "*.cfg.out" -exec rm -f '{}' ';'
    find $target_path -iname "debug*" -exec rm -rf '{}' ';' 2>/dev/null
    find $target_path -empty -type d -exec rm -f -r '{}' ';' 2>/dev/null
    find $target_path -empty -type d -exec rm -f -r '{}' ';' 2>/dev/null
fi
