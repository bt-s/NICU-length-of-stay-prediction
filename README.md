# NICU-length-of-stay-prediction
Master Thesis: Predicting the Length of Stay of Newborns in the Neonatal Intensive Care Unit

### Link to paper
- To be provided.

### Abstract

Recent advancements in machine learning and the widespread adoption of electronic health
records have enabled breakthroughs for several predictive modelling tasks in health care.
One such task that has seen considerable improvements brought by deep neural networks is
length of stay (LOS) prediction, in which research has mainly focused on adult patients in
the intensive care unit. This thesis uses multivariate time series extracted from the publicly
available Medical Information Mart for Intensive Care III database to explore the potential of
deep learning for classifying the remaining LOS of newborns in the neonatal intensive care unit
(NICU) at each hour of the stay. To investigate this, this thesis describes experiments conducted
with various deep learning models, including long short-term memory cells, gated recurrent
units, fully-convolutional networks and several composite networks. This work demonstrates
that modelling the remaining LOS of newborns in the NICU as a multivariate time series
classification problem naturally facilitates repeated predictions over time as the stay progresses
and enables advanced deep learning models to outperform a multinomial logistic regression
baseline trained on hand-crafted features. Moreover, it shows the importance of the newborn’s
gestational age and binary masks indicating missing values as variables for predicting the
remaining LOS.

### Scripts

This repository contains various scripts. This is a short explanation of their purpose:

`/scripts`

- `/create_datasets.sh'\
  This script processes data from various MIMCI-III tables to create a multivariate time series consisting of the variables:
  - Bilirubin (direct)
  - Bilirubin (indirect)
  - Blood pressure (diastolic)
  - Blood pressure (systolic)
  - Capillary refill rate
  - Fraction inspired oxygen
  - Gestational age
  - Heart rate
  - Height
  - Oxygen saturation
  - pH
  - Respiratory rate
  - Temperature
  - Weight
  
  The script selects the relevant events for each NICU subject from the relevant tables in the MIMIC-III database. The events are validated and cleaned, after which they are transformed to a multivariate time series comprising of one hour intervals. Missing values at a given time *t* are imputed through forward-filling the value at time *t-1*; if *t=0*, the imputed value is a pre-computed 'normal' value. The set of multivariate time series is split into three subsets: 64% are revserved for training, 16% are set aside for validation, and the final 20% are hold out for model evaluation. After splitting the data set, all data are normalized using normalization statistics computed on the test set. 
  
- `/modelling`\
  Please refer to the paper for the detailed information on the various networks.
  
  - `/fcn`\
  Contains scripts for training and evaluating a fully convolutional network.
  - `/gru`\
  Contains scripts for training and evaluating a gated recurrent unit architecture.
  - `/gru_cw`\
  Contains scripts for training and evaluating an architecture of channel-wise gated recurrent units.
  - `/linear_regression`\
  Contains scripts for training and evaluating a linear regression model.
  - `/logistic_regression`\
  Contains scripts for training and evaluating a logistic regression model.
  - `/lstm`\
  Contains scripts for training and evaluating a long short-term memory architecture.
  - `/lstm_cw`\
  Contains scripts for training and evaluating an architecture of channel-wise long short-term memory cells.
  - `/lstm_fcn`\
  Contains scripts for training and evaluating a composite architecture of a long short-term memory cell and a fully convolutional network.
  - `/naive_baselines`\
  Contains scripts for evaluating a naive baseline.

