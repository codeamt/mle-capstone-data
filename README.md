<h1 align="center">Generating COVIDx Dataset</h1>

<p align="center">
 Data preprocessing submodule for Udacity's Machine Learning Engineer Nanodegree program.
 <img src="https://drive.google.com/uc?export=view&id=1V4IKU42_zXyuEk_cPLx8FZdVfV4_Yd7b" width="500" />
</p>

 Generates the latest COVIDx Dataset for modeling; from benchmark research model first presented in [[1]](). 
 
 ## Repo Contents
 
<img src="https://drive.google.com/uc?export=view&id=1Lyujfg_aJv9b3budsX_q336wNKv6JCc6" width="500" />
1 directory, 6 files

## Generating Covidx Training Set 

There are 2 ways to generate the COVIDx Dataset: 

- The [data preprocessing notebook](https://github.com/codeamt/mle-capstone-data/blob/master/data_pre-processing.ipynb) (In Jupyter or Colab) 
- The [command-line tool](https://github.com/codeamt/mle-capstone-data/tree/master/data-cli-tool)

### The Data Pre-Processing Notebook: 
The data preprocessing notebook [covidnet_data_processing.ipynb](https://github.com/codeamt/mle-capstone-data/blob/master/covidnet_data_processsing.ipynb) in this repo includes additional steps for generating .csv labeling files for modeling.

### Setting up and Running data-cli-tool:

#### What you'll need: 

- Linux-based system with Python 3.7+ installed
- And/or virtualenv intalled 
- A Kaggle Authentication Key (kaggle.json file)

#### Running Locally (Linux): 

In a terminal, get the repo via git if you don't have it on your system already, then change into the repo, create a virtual environment and activate, and run the python script: 

```
pip3 install virtualenv
git clone https://github.com/codeamt/mle-capstone-data.git
cd mle-capstone-data-master && virtualenv .
source bin/activate
python3 get_covidx.py --kaggle_file "/path/to/your/kaggle.json"
```

Be sure to upload and extract the output zip file of this pipeline phase to the environment/notebook you use for the [modeling phase](https://github.com/codeamt/mle-capstone-modeling).


## About the Data 

This set aggregates and deduplicates examples to construct COVIDxv3 from the following sources:  

- https://github.com/ieee8023/covid-chestxray-dataset
- https://github.com/agchung/Figure1-COVID-chestxray-dataset
- https://github.com/agchung/Actualmed-COVID-chestxray-dataset
- https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
- https://www.kaggle.com/c/rsna-pneumonia-detection-challenge (which came from: https://nihcc.app.box.com/v/ChestXray-NIHCC)

For more notes on previous versions of the dataset, please refer to the original [COVID-Net](https://github.com/lindawangg/COVID-Net) repo for more detailed [documentation](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md). 

### Chest Radiography Images Distribution 

<img src="https://drive.google.com/uc?export=view&id=1hUFeA396Mu4k6ss6hzcMIdHylrp-9E0N" width="70%" />

[1] L. Wang and A. Wong, “COVID-Net: A Tailored Deep Convolutional Neural Network Design for Detection of COVID19 Cases from Chest Radiography Images,” ArXiv200309871 Cs Eess, Mar. 2020 [Online]. Available: http://arxiv.org/abs/2003.09871.


