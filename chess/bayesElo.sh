#! /bin/bash

# calls bayeselo with an input file
# http://www.open-aurec.com/wbforum/viewtopic.php?t=49535

if test $# -ne 1; then
   echo "This script requires one pgn file as parameter"
   exit 1
else
   params=/tmp/beparams
   echo "readpgn" $1 >$params
   echo "elo" >>$params
   echo "mm" >>$params
   echo "exactdist" >>$params
   echo "ratings >$1.ratings" >>$params
   echo "los >$1.los" >>$params
   echo "details >$1.details" >>$params
   echo "x" >>$params
   echo "x" >>$params
   bayeselo <$params >/dev/null 2>/dev/null # beyeselo's output is irrelevant here
   cat $1.ratings
fi