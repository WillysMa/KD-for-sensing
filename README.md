# Knowledge Distillation (KD) for Collaborative Learning in Distributed Communications and Sensing


## 1) Clone the repo (or download and unzip manually)
git clone <your-repo-url>
cd <your-repo-folder>

## 2) Create the dataset directory
New-Item -ItemType Directory -Force -Path dataset\scenario9 | Out-Null

## 3) (Manual step) Download Scenario 9 from:
 https://www.deepsense6g.net/scenarios/Scenarios%201-9/scenario-9
### Extract it so you have:
 dataset\scenario9\unit1\
 dataset\scenario9\scenario9.csv

## 4) Verify the expected files exist
Get-ChildItem dataset\scenario9
Get-ChildItem dataset\scenario9\unit1 | Select-Object -First 5

## 5) Run the preprocessing scripts (in order)
python CSV_process.py
python gen_data_seq.py

## 6) Confirm the two output CSVs were created
Get-ChildItem dataset\scenario9\*.csv
 
## Explanation:
-- run train.py to train the teacher model without KD

-- run train_SelfKD.py to train the teacher model with self-KD

-- run train_RKD.py to train the student model:

1) kd_mode=0: no KD 2) kd_mode=1: conventional KD 3) kd_mode=5: relational KD

## Dataset
We use the Deepsense 6G, scenario 9 for trianing and testing.

## Models and hyperparameters:
Five models contained: 
1) Teacher model without KD
2) Teacher model with self-KD 
3) Student model without KD
4) Student model with conventional KD
5) Student model with relational KD
   
The hyperparameters are shown in the txt files.
