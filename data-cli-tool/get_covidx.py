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
#dest = os.path.join(os.environ['KAGGLE_CONFIG_DIR'], "kaggle.json")

#shutil.copy(kaggle_file, dest)
#os.chmod(dest, 600)

with open(str(kaggle_file)) as f:
    print(str(kaggle_file))
    keys = json.load(f)
    os.environ["KAGGLE_USERNAME"] = keys["username"]
    os.environ["KAGGLE_KEY"] = keys["key"]
print("Done Configuring Kaggle.")


### Download data from Kaggle + GitHub ###
print("Downloading Data from Kaggle and GitHub...")
subprocess.call(["kaggle", "datasets", "download", "-d", "tawsifurrahman/covid19-radiography-database", "--force"])
subprocess.call(["wget", "--no-check-certificate", "-r", "https://docs.google.com/uc?export=download&id=1zeyP13qKxNS78ML-cdG4M2SO89k3j4NG", "-O", "COVID-19 Radiography Database.zip"])
subprocess.call(["kaggle", "competitions", "download", "-c", "rsna-pneumonia-detection-challenge"])
subprocess.call(["git","clone", "https://github.com/ieee8023/covid-chestxray-dataset.git"])
subprocess.call(["git","clone", "https://github.com/agchung/Figure1-COVID-chestxray-dataset"])
subprocess.call(["git","clone", "https://github.com/agchung/Actualmed-COVID-chestxray-dataset"])


### Extract ###

os.mkdir('rsna-data')
rsna_zip = zipfile.ZipFile("./rsna-pneumonia-detection-challenge.zip")
rsna_zip.extractall('rsna-data')
rsna_zip.close()
radio_db_zip = zipfile.ZipFile("./COVID-19 Radiography Database.zip")
radio_db_zip.extractall()
covid19_radiography_db = zipfile.ZipFile("./covid19-radiography-database.zip")
covid19_radiography_db.extractall()
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
print("Seeding; Storing References to Needed File Paths...")
# set parameters here
savepath = './data/data'
seed = 0
np.random.seed(seed) # Reset the seed so all runs are the same.
random.seed(seed)
MAXVAL = 255  # Range [0 255]
# path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
cohen_imgpath = './covid-chestxray-dataset/images'
cohen_csvpath = './covid-chestxray-dataset/metadata.csv'
# path to covid-19 dataset from https://github.com/agchung/Figure1-COVID-chestxray-dataset
fig1_imgpath = './Figure1-COVID-chestxray-dataset/images'
fig1_csvpath = './Figure1-COVID-chestxray-dataset/metadata.csv'
# path to covid-19 dataset from https://github.com/agchung/Actualmed-COVID-chestxray-dataset
actmed_imgpath = './Actualmed-COVID-chestxray-dataset/images'
actmed_csvpath = './Actualmed-COVID-chestxray-dataset/metadata.csv'
# path to covid-19 dataset from https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
sirm_imgpath = './COVID-19 Radiography Database/COVID-19'
sirm_csvpath = './COVID-19 Radiography Database/COVID-19.metadata.xlsx'
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

### PROCESS COHEN FILES ###
print("Now Processing Databases...")
# to avoid duplicates
patient_imgpath = {}
# adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py#L814
cohen_csv = pd.read_csv(cohen_csvpath, nrows=None)
#idx_pa = csv["view"] == "PA"  # Keep only the PA view
views = ["PA", "AP", "AP Supine", "AP semi erect", "AP erect"]
cohen_idx_keep = cohen_csv.view.isin(views)
cohen_csv = cohen_csv[cohen_idx_keep]
fig1_csv = pd.read_csv(fig1_csvpath, encoding='ISO-8859-1', nrows=None)
actmed_csv = pd.read_csv(actmed_csvpath, nrows=None)
sirm_csv = pd.read_excel(sirm_csvpath)
# get non-COVID19 viral, bacteria, and COVID-19 infections from covid-chestxray-dataset, figure1 and actualmed
# stored as patient id, image filename and label
filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
covid_ds = {'cohen': [], 'fig1': [], 'actmed': [], 'sirm': []}
print("Working on Cohen Database...")
for index, row in cohen_csv.iterrows():
  f = row['finding'].split(',')[0] # take the first finding, for the case of COVID-19, ARDS
  if f in mapping:
    count[mapping[f]] += 1
    entry = [str(row['patientid']), row['filename'], mapping[f], 'cohen']
    filename_label[mapping[f]].append(entry)
    if mapping[f] == 'COVID-19':
      covid_ds['cohen'].append(str(row['patientid']))
print("Done.")
print("Working on covid-chestxray-dataset Database...")
for index, row in fig1_csv.iterrows():
  if not str(row['finding']) == 'nan':
    f = row['finding'].split(',')[0] # take the first finding
    if f in mapping:
      count[mapping[f]] += 1
      if os.path.exists(os.path.join(fig1_imgpath, row['patientid'] + '.jpg')):
        entry = [row['patientid'], row['patientid'] + '.jpg', mapping[f], 'fig1']
      elif os.path.exists(os.path.join(fig1_imgpath, row['patientid'] + '.png')):
        entry = [row['patientid'], row['patientid'] + '.png', mapping[f], 'fig1']
      filename_label[mapping[f]].append(entry)
      if mapping[f] == 'COVID-19':
        covid_ds['fig1'].append(row['patientid'])
print("Done.")
print("Working on Actualmed Database...")
for index, row in actmed_csv.iterrows():
  if not str(row['finding']) == 'nan':
    f = row['finding'].split(',')[0]
    if f in mapping:
      count[mapping[f]] += 1
      entry = [row['patientid'], row['imagename'], mapping[f], 'actmed']
      filename_label[mapping[f]].append(entry)
      if mapping[f] == 'COVID-19':
        covid_ds['actmed'].append(row['patientid'])
print("Done.")
print("Cleaning/Deduplicating...")
sirm = set(sirm_csv['URL'])
cohen = set(cohen_csv['url'])
discard = ['100', '101', '102', '103', '104', '105',
           '110', '111', '112', '113', '122', '123',
           '124', '125', '126', '217']
print("Done.")
print("Working on COVID-19 Radiography Database...")
for idx, row in sirm_csv.iterrows():
  patientid = row['FILE NAME']
  if row['URL'] not in cohen and patientid[patientid.find('(')+1:patientid.find(')')] not in discard:
    count[mapping['COVID-19']] += 1
    imagename = patientid + '.' + row['FORMAT'].lower()
    if not os.path.exists(os.path.join(sirm_imgpath, imagename)):
      imagename = patientid.split('(')[0] + ' ('+ patientid.split('(')[1] + '.' + row['FORMAT'].lower()
      entry = [patientid, imagename, mapping['COVID-19'], 'sirm']
      filename_label[mapping['COVID-19']].append(entry)
      covid_ds['sirm'].append(patientid)
print("Done.")
print(f'Data distribution from covid datasets: {count}')


# add covid-chestxray-dataset, figure1 and actualmed into COVIDx dataset
# since these datasets don't have test dataset, split into train/test by patientid
# for covid-chestxray-dataset:
# patient 8 is used as non-COVID19 viral test
# patient 31 is used as bacterial test
# patients 19, 20, 36, 42, 86 are used as COVID-19 viral test
# for figure 1:
# patients 24, 25, 27, 29, 30, 32, 33, 36, 37, 38

#### DISTRIBUTE COHEN AND FIG1 BASED ON TRAIN/TEST SPLIT ####
print("Assigning current patient IDs to either Train/Test...")
ds_imgpath = {'cohen': cohen_imgpath, 'fig1': fig1_imgpath, 'actmed': actmed_imgpath, 'sirm': sirm_imgpath}
for key in filename_label.keys():
  arr = np.array(filename_label[key])
  if arr.size == 0:
    continue
  # split by patients
  # num_diff_patients = len(np.unique(arr[:,0]))
  # num_test = max(1, round(split*num_diff_patients))
  # select num_test number of random patients
  # random.sample(list(arr[:,0]), num_test)
  if key == 'pneumonia':
    test_patients = ['8', '31']
  elif key == 'COVID-19':
    test_patients = ['19', '20', '36', '42', '86',
    '94', '97', '117', '132', '138', '144', '150', '163', '169', '174', '175', '179', '190', '191','COVID-00024', 'COVID-00025', 'COVID-00026', 'COVID-00027', 'COVID-00029','COVID-00030', 'COVID-00032', 'COVID-00033', 'COVID-00035', 'COVID-00036','COVID-00037', 'COVID-00038','ANON24', 'ANON45', 'ANON126', 'ANON106', 'ANON67','ANON153', 'ANON135', 'ANON44', 'ANON29', 'ANON201', 'ANON191', 'ANON234', 'ANON110', 'ANON112', 'ANON73','ANON220', 'ANON189', 'ANON30', 'ANON53', 'ANON46','ANON218', 'ANON240', 'ANON100', 'ANON237', 'ANON158','ANON174', 'ANON19', 'ANON195','COVID-19(119)', 'COVID-19(87)', 'COVID-19(70)', 'COVID-19(94)','COVID-19(215)', 'COVID-19(77)', 'COVID-19(213)', 'COVID-19(81)', 'COVID-19(216)', 'COVID-19(72)', 'COVID-19(106)', 'COVID-19(131)', 'COVID-19(107)', 'COVID-19(116)', 'COVID-19(95)', 'COVID-19(214)',
                         'COVID-19(129)']
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
      if patient[3] == 'sirm':
        image = cv2.imread(os.path.join(ds_imgpath[patient[3]], patient[1]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        patient[1] = patient[1].replace(' ', '')
        cv2.imwrite(os.path.join(savepath, 'test', patient[1]), gray)
      else:
        copyfile(os.path.join(ds_imgpath[patient[3]], patient[1]), os.path.join(savepath, 'test', patient[1]))
      test.append(patient)
      test_count[patient[2]] += 1
    else:
      if patient[3] == 'sirm':
        image = cv2.imread(os.path.join(ds_imgpath[patient[3]], patient[1]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        patient[1] = patient[1].replace(' ', '')
        cv2.imwrite(os.path.join(savepath, 'train', patient[1]), gray)
      else:
        copyfile(os.path.join(ds_imgpath[patient[3]], patient[1]), os.path.join(savepath, 'train', patient[1]))
      train.append(patient)
      train_count[patient[2]] += 1
print("Done.")
print(f'test count: {str(test_count)}')
print(f'train count: {str(train_count)}')

### PROCESS RSNA DB FILES ###
# add normal and rest of pneumonia cases from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
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
    ds = dicom.dcmread(os.path.join(rsna_datapath, rsna_imgpath, patient + '.dcm'))
    pixel_array_numpy = ds.pixel_array
    imgname = patient + '.png'
    if patient in test_patients:
      cv2.imwrite(os.path.join(savepath, 'test', imgname), pixel_array_numpy)
      test.append([patient, imgname, key, 'rsna'])
      test_count[key] += 1
    else:
      cv2.imwrite(os.path.join(savepath, 'train', imgname), pixel_array_numpy)
      train.append([patient, imgname, key, 'rsna'])
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
print("Writing Metadata to .txt files (training)...")
train_file = open("train_split_v3.txt",'w')
for sample in train:
  if len(sample) == 4:
    info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
  else:
    info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
  train_file.write(info)
train_file.close()
print("Done.")
print("Writing Metadata to .txt files (testing)...")
test_file = open("test_split_v3.txt", 'w')
for sample in test:
  if len(sample) == 4:
    info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
  else:
    info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
  test_file.write(info)
test_file.close()
print("Done.")
print("Writing Metadata to .txt files to .csv files...")
trainset_meta_txt = './train_split_v3.txt'
testset_meta_txt = './test_split_v3.txt'
trainset_meta_csv = 'train_split_v3.csv'
testset_meta_csv = 'test_split_v3.csv'
column_names = ['patient_id', 'fname', 'label', "label_short"]
train_df = pd.read_csv(trainset_meta_txt, delimiter=' ', names=column_names, index_col=False)
test_df = pd.read_csv(testset_meta_txt, delimiter=' ', names=column_names, index_col=False)
print(f"Training Set Count by Label: \n{train_df.groupby('label').count()}")
print(f"Testing Set Count by Label: \n{test_df.groupby('label').count()}")
train_df.to_csv(trainset_meta_csv, index=False, sep = ',')
test_df.to_csv(testset_meta_csv, index=False, sep = ',')

#### Move labels to working dir for data modeling ###
shutil.move("./train_split_v3.csv", savepath)
shutil.move("./test_split_v3.csv", savepath)
print("Done.")

#### Create Archive (.zip) for modeling ###
print("Creating archive....")
shutil.make_archive('data', 'zip', 'data')
print("Done. Data zip for training located at: './data.zip'",)

#### Cleanup/Reset ###
print("Cleaning Up...")
os.remove("./rsna-pneumonia-detection-challenge.zip")
os.remove("./train_split_v3.txt")
os.remove("./test_split_v3.txt")
os.remove("./COVID-19 Radiography Database.zip")
shutil.rmtree("./covid-chestxray-dataset")
shutil.rmtree("./Actualmed-COVID-chestxray-dataset")
shutil.rmtree("./COVID-19 Radiography Database/")
shutil.rmtree("./Figure1-COVID-chestxray-dataset")
shutil.rmtree("./rsna-data")
shutil.rmtree("./data")
os.mkdir("data")
os.mkdir("data/train")
os.mkdir("data/test")
end = time()
process_time = end - start
print(f"Data Generation Complete. Processing Time: {str(process_time/60)} minutes.")
