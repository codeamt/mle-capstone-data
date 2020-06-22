import os
import sys
from time import time
import json
import argparse
import numpy as np
import pandas as pd
import random
import shutil
from shutil import copyfile
import zipfile
import pydicom as dicom
import cv2
import subprocess
from pathlib import Path


start = time()

##### GET ARGS #####
parser = argparse.ArgumentParser(description='Generates Covidx Dataset')
parser.add_argument('--kaggle_file',
                       type=Path,
                       help='path to your kaggle.json file')
args = parser.parse_args()
kaggle_file = args.kaggle_file

#### SETUP KAGGLE #####
print("Configuring Kaggle...")

if not os.path.exists(kaggle_file):
  print('A kaggle file does not exist at this location:')
  print('\n'.join(aggle_file))
  sys.exit()

os.environ['KAGGLE_CONFIG_DIR'] = "."
with open(str(kaggle_file)) as f:
    print(str(kaggle_file))
    keys = json.load(f)
    os.environ["KAGGLE_USERNAME"] = keys["username"]
    os.environ["KAGGLE_KEY"] = keys["key"]
print("Done Configuring Kaggle.")


### Download data from Kaggle + GitHub ###
print("Downloading Data from Kaggle and GitHub...")
subprocess.call(["kaggle", "competitions", "download", "-c", "rsna-pneumonia-detection-challenge"])
subprocess.call(["git","clone", "https://github.com/ieee8023/covid-chestxray-dataset.git"])
subprocess.call(["git","clone", "https://github.com/agchung/Figure1-COVID-chestxray-dataset"])
os.mkdir('rsna-data')
rsna_zip = zipfile.ZipFile("./rsna-pneumonia-detection-challenge.zip")
rsna_zip.extractall('rsna-data')
rsna_zip.close()
print("Done Downloading Data.")

### Set default values ###
print("Setting seeds and splitting data")
seed = 0
np.random.seed(seed)
random.seed(seed)
MAXVAL = 255
split = 0.1 # train/test split
print(f'Done. Train/Test Split: {str(split)}')

###### RELEVANT FILE PATHS ######
print("Storing References to Needed File Paths...")
savepath = "./data"
# path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
cohen_imgpath = './covid-chestxray-dataset/images'
cohen_csvpath = './covid-chestxray-dataset/metadata.csv'
# path to covid-14 dataset from https://github.com/agchung/Figure1-COVID-chestxray-dataset
fig1_imgpath = './Figure1-COVID-chestxray-dataset/images'
fig1_csvpath = './Figure1-COVID-chestxray-dataset/metadata.csv'
# path to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
rsna_datapath = './rsna-data'
# get all the normal from here
rsna_csvname = 'stage_2_detailed_class_info.csv'
# get all the 1s from here since 1 indicate pneumonia
# found that images that aren't pneunmonia and also not normal are classified as 0s
rsna_csvname2 = 'stage_2_train_labels.csv'
rsna_imgpath = 'stage_2_train_images'
print("References Stored.")


# parameters for COVIDx dataset
print("Setting Up Counters...")
train = []
test = []
test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
print("Done Setting Up Counters.")

### MAP NON-COVID, NON-NORMAL FILES TO PNEUMONIA ###
print("Creating A Label Map. Non-COVID-19 infections all map to Pneumonia...")
mapping = dict()
mapping['COVID-19'] = 'COVID-19'
mapping['SARS'] = 'pneumonia'
mapping['MERS'] = 'pneumonia'
mapping['Streptococcus'] = 'pneumonia'
mapping['Klebsiella'] = 'pneumonia'
mapping['Chlamydophila'] = 'pneumonia'
mapping['Legionella'] = 'pneumonia'
mapping['Normal'] = 'normal'
mapping['Lung Opacity'] = 'pneumonia'
mapping['1'] = 'pneumonia'
print("Done Creating Label Map.")


### PATIENT OBJECT to avoid duplicates ###
patient_imgpath = {}
# adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py#L814

### PROCESS COHEN FILES ###
print("Now Processing Cohen Database...")
cohen_csv = pd.read_csv(cohen_csvpath, nrows=None)
views = ["PA", "AP", "AP Supine", "AP semi erect", "AP erect"]
cohen_idx_keep = cohen_csv.view.isin(views)
cohen_csv = cohen_csv[cohen_idx_keep]
fig1_csv = pd.read_csv(fig1_csvpath, encoding='ISO-8859-1', nrows=None)
# get non-COVID19 viral, bacteria, and COVID-19 infections from covid-chestxray-dataset
# stored as patient id, image filename and label
filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
for index, row in cohen_csv.iterrows():
    f = row['finding'].split(',')[0] # take the first finding, for the case of COVID-19, ARDS
    if f in mapping: #
        count[mapping[f]] += 1
        entry = [str(row['patientid']), row['filename'], mapping[f], row['view']]
        filename_label[mapping[f]].append(entry)
print("Done.")

### PROCESS FIG1 FILES ###
print("Now Processing covid-chestxray-dataset...")
for index, row in fig1_csv.iterrows():
    if not str(row['finding']) == 'nan':
        f = row['finding'].split(',')[0] # take the first finding
        if f in mapping: #
            count[mapping[f]] += 1
            if os.path.exists(os.path.join(fig1_imgpath, row['patientid'] + '.jpg')):
                entry = [row['patientid'], row['patientid'] + '.jpg', mapping[f]]
            elif os.path.exists(os.path.join(fig1_imgpath, row['patientid'] + '.png')):
                entry = [row['patientid'], row['patientid'] + '.png', mapping[f]]
            filename_label[mapping[f]].append(entry)
print("Done")
print(f'Data distribution: {count}')


#### DISTRIBUTE COHEN AND FIG1 BASED ON TRAIN/TEST SPLIT ####
print("Assigning Patient IDs to either Train/Test...")
for key in filename_label.keys():
    arr = np.array(filename_label[key])
    if arr.size == 0:
        continue
    # split by patients
    # num_diff_patients = len(np.unique(arr[:,0]))
    # num_test = max(1, round(split*num_diff_patients))
    # select num_test number of random patients
    if key == 'pneumonia':
        test_patients = ['8', '31']
    elif key == 'COVID-19':
        test_patients = ['19', '20', '36', '42', '86',
                         '94', '97', '117', '132',
                         '138', '144', '150', '163', '169'] # random.sample(list(arr[:,0]), num_test)
    else:
        test_patients = []
    print(f'Key: {key}')
    print(f'Test patients: {test_patients}')
    # go through all the patients
    for patient in arr:
        if patient[0] not in patient_imgpath:
            patient_imgpath[patient[0]] = [patient[1]]
        else:
            if patient[1] not in patient_imgpath[patient[0]]:
                patient_imgpath[patient[0]].append(patient[1])
            else:
                continue  # skip since image has already been written
        if patient[0] in test_patients:
            copyfile(os.path.join(cohen_imgpath, patient[1]), os.path.join(savepath, 'test', patient[1]))
            test.append(patient)
            test_count[patient[2]] += 1
        else:
            if 'COVID' in patient[0]:
                copyfile(os.path.join(fig1_imgpath, patient[1]), os.path.join(savepath, 'train', patient[1]))
            else:
                copyfile(os.path.join(cohen_imgpath, patient[1]), os.path.join(savepath, 'train', patient[1]))
            train.append(patient)
            train_count[patient[2]] += 1
print("Done.")
print(f'test count: {str(test_count)}')
print(f'train count: {str(train_count)}')



### PROCESS RSNA DB FILES ###
print("Now Processing RSNA database...")
csv_normal = pd.read_csv(os.path.join(rsna_datapath, rsna_csvname), nrows=None)
csv_pneu = pd.read_csv(os.path.join(rsna_datapath, rsna_csvname2), nrows=None)
patients = {'normal': [], 'pneumonia': []}
for index, row in csv_normal.iterrows():
    if row['class'] == 'Normal':
        patients['normal'].append(row['patientId'])

for index, row in csv_pneu.iterrows():
    if int(row['Target']) == 1:
        patients['pneumonia'].append(row['patientId'])

for key in patients.keys():
    arr = np.array(patients[key])
    if arr.size == 0:
        continue
    # split by patients
    # num_diff_patients = len(np.unique(arr))
    # num_test = max(1, round(split*num_diff_patients))
    test_patients = np.load('./assets/rsna_test_patients_{}.npy'.format(key)) # random.sample(list(arr), num_test), download the .npy files from the repo.
    # np.save('rsna_test_patients_{}.npy'.format(key), np.array(test_patients))
    for patient in arr:
        if patient not in patient_imgpath:
            patient_imgpath[patient] = [patient]
        else:
            continue  # skip since image has already been written

        #### DISTRIBUTE RSNA IMGS BASED ON TRAIN/TEST SPLIT ####
        ds = dicom.dcmread(os.path.join(rsna_datapath, rsna_imgpath, patient + '.dcm'))
        pixel_array_numpy = ds.pixel_array
        imgname = patient + '.png'
        if patient in test_patients:
            cv2.imwrite(os.path.join(savepath, 'test', imgname), pixel_array_numpy)
            test.append([patient, imgname, key])
            test_count[key] += 1
        else:
            cv2.imwrite(os.path.join(savepath, 'train', imgname), pixel_array_numpy)
            train.append([patient, imgname, key])
            train_count[key] += 1

print("Done. All Files Loaded")
print(f'test count: {str(test_count)}')
print(f'train count: {str(train_count)}')
print('Final stats')
print('Train count: ', train_count)
print('Test count: ', test_count)
print('Total length of train: ', len(train))
print('Total length of test: ', len(test))



### GENERATE CSV LABEL FILES FOR TRAINING ###
# export to train and test csv
# format as patientid, filename, label, separated by a space
print("Writing Metadata to .csv files for training...")
trainset_meta_txt = './assets/train_COVIDx3.txt'
testset_meta_txt = './assets/test_COVIDx3.txt'
train_file = open(trainset_meta_txt,"a")
for sample in train:
    if len(sample) == 4:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
    else:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
    train_file.write(info)

train_file.close()
test_file = open(testset_meta_txt, "a")
for sample in test:
    if len(sample) == 4:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
    else:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
    test_file.write(info)
test_file.close()
trainset_meta_csv = 'train_split_v3.csv'
testset_meta_csv = 'test_split_v3.csv'
column_names = ['patient_id', 'fname', 'label', "label_short"]
train_df = pd.read_csv(trainset_meta_txt, delimiter=' ', names=column_names, index_col=False)
test_df = pd.read_csv(testset_meta_txt, delimiter=' ', names=column_names, index_col=False)
train_df.to_csv(trainset_meta_csv, index=False, sep = ',')
test_df.to_csv(testset_meta_csv, index=False, sep = ',')

#### Move labels to working dir for data modeling ###
shutil.move("./train_split_v3.csv", savepath)
shutil.move("./test_split_v3.csv", savepath)
print("Done.")
print("Creating archive....")
shutil.make_archive('data', 'zip', 'data')
print("Done. Data zip for training located at: './data.zip'",)

#### Move labels to working dir for data modeling ###
print("Cleaning Up...")
os.remove("./rsna-pneumonia-detection-challenge.zip")
shutil.rmtree("./covid-chestxray-dataset")
shutil.rmtree("./Figure1-COVID-chestxray-dataset")
shutil.rmtree("./rsna-data")
end = time()
process_time = end - start
print(f"Data Generation Complete. Processing Time: {str(process_time)} seconds.")
