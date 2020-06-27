<h1 align="center">Generating COVIDx Dataset</h1>

<p align="center">
 Data preprocessing submodule for Udacity's Machine Learning Engineer Nanodegree program. \n
 <img src="https://drive.google.com/uc?export=view&id=1xE3OFoQP3hdyI8nP_iU_i9Y7zLJZ4QQO" width="300" />
</p>
 
 ## Repo Contents
 
<img src="https://drive.google.com/uc?export=view&id=1oX3lTBcAGrZcfSB-ZyIs5D3wymxcPToF" />
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


Be sure to upload and extract the output zip file of this notebook in your modeling environment.


## About the Data 

This generates training and test samples for COVIDxv3, the dataset used in [1]. For more notes on previous versions, please refer to the COVID-Net repo. For detailed instructions on how to generate the COVIDx dataset, visit the [original repo](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md) for detailed instructions. 

### Chest Radiography Images Distribution 

<img src="https://drive.google.com/uc?export=view&id=1IhjhezM8GYKbUQPeHRTQsFVKW-NE6OHk" width="50%" />


