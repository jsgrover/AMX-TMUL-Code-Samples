#!/usr/bin/bash


if [ -z $1 ] ; then
	echo "Usage : covrun.sh <proj>"
	exit 1;
fi;

BUILDDIR=$1

make clean

cov-configure --g++
cov-build --dir BUILDDIR make 
cov-analyze --dir BUILDDIR
cov-format-errors --dir BUILDDIR --html-output htmlResults

