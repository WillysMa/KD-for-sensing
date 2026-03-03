# Knowledge Distillation (KD) for Collaborative Learning in Distributed Communications and Sensing

## Dataset Preparation (Step 1)

1. Download the project and extract it to your local machine.

2. Download **Scenario 9** from:  
   https://www.deepsense6g.net/scenarios/Scenarios%201-9/scenario-9

3. Extract the dataset to form the file structure:

```text
dataset/
└── scenario9/
    ├── unit1/
    └── scenario9.csv
 ```
5. Run the preprocessing scripts CSV_process.py and gen_data_seq.py in order

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
