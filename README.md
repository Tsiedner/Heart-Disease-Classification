### Predicting heart disease using Machine Learning (ML)
This notebook looks into using various Python based ML and data science libraries in an attempt to build a ML learning model capable of predicting whether or not someone has heart disease based on their medical attributes.

The approach that will be taken:

  1. Data
  2. Evaluation
  3. Features
  4. Modelling
  5. Experimentation
  
## 1. Problem Definition

Given clinical parameter about a patient, can we predict whether or not they have heart disease?

## 2. Data

The original data came from the Cleavland from the UCI Machine Learning Repository.

## 3. Evaluation

If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.

## 4. Features

# Create data dictionary

age - age in years

sex - (1 = male; 0 = female)

cp - chest pain type

  1. Typical angina: chest pain related decrease blood supply to the heart
  2. Atypical angina: chest pain not related to heart
  3. Non-anginal pain: typically esophageal spasms (non heart related)
  4. Asymptomatic: chest pain not showing signs of disease
  
trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern

chol - serum cholestoral in mg/dl serum = LDL + HDL + .2 * triglycerides above 200 is cause for concern

fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) '>126' mg/dL signals diabetes

restecg - resting electrocardiographic results

  1. Nothing to note
  2. ST-T Wave abnormality can range from mild symptoms to severe problems signals non-normal heart beat
  3. Possible or definite left ventricular hypertrophy Enlarged heart's main pumping chamber
  
thalach - maximum heart rate achieved

exang - exercise induced angina (1 = yes; 0 = no)

oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more

slope - the slope of the peak exercise ST segment

  1. Upsloping: better heart rate with excercise (uncommon)
  2. Flatsloping: minimal change (typical healthy heart)
  3. Downslopins: signs of unhealthy heart
  
ca - number of major vessels (0-3) colored by flourosopy colored vessel means the doctor can see the blood passing through the more blood movement the better (no clots)

thal - thalium stress result 1

  1. normal
  2. fixed defect: used to be defect but ok now
  3. reversable defect: no proper blood movement when excercising
  
target - have disease or not (1=yes, 0=no) (= the predicted attribute)
