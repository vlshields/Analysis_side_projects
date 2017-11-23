# Predict "No-shows" at Doctors office

The file doctor_visits.py contains my attempt at creating a predictive model for wether or not a patient will show up to their doctors appointment. The data was taken from kaggle.com, and contains data medical appointments, of the public healthcare of the capital city of Espirito Santo State - Vitoria - Brazil. The findings are not externally valid for other healthcare systems.

## Usage

Install the requirements.
```
pip install -r requirements.txt
```

Run the program
```
python3 doctor_visits.py
```

# Conclusions

The dataset is imbalanced (there are far more show ups that no-shows), which makes it difficult to build a predictive model that predicts no-shows very well. The model is decently good at predicticting show ups, but thats likely because most of the data are show ups. The test set accuracy is around 80 percent, but 79 percent of the data do show up, so its hard to say if the model is learning anything. A confusion matrix is included for reference.