# TC_calculation
AM alloys thermal conductivity (TC) dataset is created by measuring the TC of various alloys using Netzsch LFA 467 HT HyperFlash®- light flash apparatus.
At first, machine learning models with default parameters are trained with the datset and the optimal model machine learning model is selected based on the statistical metrices. The code of training all model is inside Five model performance.
The optimal model is futher tuned up using GRID search method.The overall code is located inside hypertunning the optimal model.py file
The final otimal model is also tested with the ICONEL 718 AM alloys and good aggrement between machine learning model and experiments.
The GRCop alloys also has 72 total data and hypertunning was done using Bayesian. Bayesian is hypertunning is very reliable when dataset is very small. The computational time is less than other hypertunning methods. The training and testing results were good. It has been again tested to unknown GRCop and has good match with the experiments. The coding file of this is located inside Bayesian for GRCop data.
The important feature rank with the help of SHAP method. The feature important calculation code lies inside Feature importance using SHAP.py file.
