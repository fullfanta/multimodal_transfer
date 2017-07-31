#!/usr/bin/env bash

# make a data folder
if ! [ -e data ]
then
    mkdir data
fi

pushd data

# download images
declare -a filenames=("train2014")
for i in "${filenames[@]}"
do
    if ! [ -e $i.zip ]
    then
        echo $i.zip "not found, downloading"
        wget http://msvocds.blob.core.windows.net/coco2014/$i.zip
    fi
    unzip $i.zip
done

popd
