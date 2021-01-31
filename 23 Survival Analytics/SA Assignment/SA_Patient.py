
import lifelines
import pandas as pd


pat = pd.read_csv("D:\\360Assignments\\Submission\\23 Survival Analytics\\SA Assignment\\Patient.csv")
pat.head()
pat.describe()

# Spell is referring to time 
pat.columns
T = pat.Followup

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed=pat.Eventtype)

# Time-line estimations plot 
kmf.plot()

pat.PatientID.value_counts()


# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(pat['Followup'], pat['Eventtype']==1, label='1')
ax = kmf.plot()


kmf.fit(pat['Followup'], pat['Eventtype']==0, label='0')
ax = kmf.plot()





