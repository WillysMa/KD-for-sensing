# Knowledge Distillation (KD) for Collaborative Learning in Distributed Communications and Sensing

## Dataset Preparation

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

## Training model:
-- run train_both.py to train model based on both image and radar data. 

-- run train_image.py to train model based on only image data. 

1) kd_mode=0: no KD 2) kd_mode=1: conventional KD 3) kd_mode=2: relational KD

## Testing model
All trained model along with the hyparameters are under the folder: All_models/
-- run test_model_both.py to test the model based on both modalities

-- run test_model_image.py to test the model based only image

### Models and hyperparameters:
Nine models contained: 
1) BothTeacher_best.pth: Best teacher model based on both modalitis
2) ImageTeacher_noKD.pth: Best image-based teacher model without self-KD refinement
3) ImageTeacher_best.pth: Best image-based teacher model with self-KD refinement
4) ImageStd_noKD.pth:  Image-based student model without KD
5) ImageStd_KD.pth: Image-based student model with conventional KD
6) ImageStd_RKD.pth: Image-based student model with relational KD
7) BothStd_noKD.pth: Student model based on both modalities without KD
8) BothStd_KD.pth: Student model based on both modalities with conventional KD
9) BothStd_RKD.pth: Student model based on both modalities with relational KD
   
The hyperparameters are shown in the txt files.
