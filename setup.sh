#!/bin/bash

# dont download the data folder if it already exists
if [ ! -d "data" ]; then
    # download the data and unzip it
    wget https://gtvault-my.sharepoint.com/:u:/g/personal/hbaldwa3_gatech_edu/EY8L0BWf6iFDjf9bcUeNkq8BoXzurFZljRwmOpajmhv5Ng?download=1 -O data.zip
    unzip data.zip
    rm data.zip
fi

# install all the dependencies
pip install -r requirements.txt
