
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('bank-additional-full.csv', delimiter = ';')
Age = dataset.iloc[:, 0:1].values
Job = dataset.iloc[:, 1:2].values
Marital = dataset.iloc[:, 2:3].values
Education = dataset.iloc[:, 3:4].values
Default = dataset.iloc[:, 4:5].values
Housing = dataset.iloc[:, 5:6].values
Loan = dataset.iloc[:, 6:7].values
Contact = dataset.iloc[:, 7:8].values
Duration = dataset.iloc[:,10:11].values

Response = dataset.iloc[:, 20:21].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

def __plot__(X, Y, label):
    #bins = np.linspace(0, np.max(X)+2, np.max(X)+2)
    plt.hist([X[Y == 1],X],alpha =1, label=["Positive Reply","All Replies"],density=True)
    plt.legend(loc = 'upper right')
    plt.savefig('Images/'+label+'.png')
    plt.show()

#All the different graphs for study    
Response = le.fit_transform(Response)
print(le.classes_)
__plot__(Age, Response, "Age-Response Graph")

Job = le.fit_transform(Job)
print(le.classes_)
__plot__(Job, Response, "Job-Response Graph")

Marital = le.fit_transform(Marital)
print(le.classes_)
__plot__(Marital, Response, "Marital-Response Graph")

Education = le.fit_transform(Education)
print(le.classes_)
__plot__(Education, Response, "Education-Response Graph")

Default = le.fit_transform(Default)
print(le.classes_)
__plot__(Default, Response, "Default-Response Graph")

Housing = le.fit_transform(Housing)
print(le.classes_)
__plot__(Housing, Response, "Housing-Response Graph")

Loan = le.fit_transform(Loan)
print(le.classes_)
__plot__(Loan, Response, "Loan-Response Graph")

Contact = le.fit_transform(Contact)
print(le.classes_)
__plot__(Contact, Response, "Contact-Response Graph")

#Analysis of All graphs
Time_Wasted = []
Failure_Rate = []
Response = np.reshape(Response,(-1,1))
Time_Wasted.append(np.sum(Duration[(((Age < 20)  + (Age > 35)) * (Age < 55))]))
Failure_Rate.append(np.sum(Duration[(Response == 0) & (((Age < 20)  + (Age > 35)) * (Age < 55))])/np.sum(Duration[(((Age < 20)  + (Age > 35)) * (Age < 55))])*100)

Job = np.reshape(Job,(-1,1))
Time_Wasted.append(np.sum(Duration[(Job != 0)  * (Job != 4) * (Job != 8)]))
Failure_Rate.append(np.sum(Duration[(Response == 0) & ((Job != 0)  * (Job != 4) * (Job != 8))])/np.sum(Duration[(Job != 0)  * (Job != 4) * (Job != 8)])*100)

Marital = np.reshape(Marital,(-1,1))
Time_Wasted.append(np.sum(Duration[(Marital == 0)  + (Marital == 3)]))
Failure_Rate.append(np.sum(Duration[(Response == 0) & ((Marital == 0)  + (Marital == 3))])/np.sum(Duration[(Marital == 0)  + (Marital == 3)])*100)

Education = np.reshape(Education,(-1,1))
Time_Wasted.append(np.sum(Duration[(Education != 3)  * (Education != 5) * (Education != 6)]))
Failure_Rate.append(np.sum(Duration[(Response == 0) & ((Education != 3)  * (Education != 5) * (Education != 6))])/np.sum(Duration[(Education != 3)  * (Education != 5) * (Education != 6)])*100)

Default = np.reshape(Default,(-1,1))
Time_Wasted.append(np.sum(Duration[(Default != 0)]))
Failure_Rate.append(np.sum(Duration[(Response == 0) & ((Default != 0))])/np.sum(Duration[(Default != 0)])*100)

Housing = np.reshape(Housing,(-1,1))
Time_Wasted.append(np.sum(Duration[(Housing == 1)]))
Failure_Rate.append(np.sum(Duration[(Response == 0) & ((Housing == 1))])/np.sum(Duration[(Housing == 1)])*100)

Loan = np.reshape(Loan,(-1,1))
Time_Wasted.append(np.sum(Duration[(Loan != 0)]))
Failure_Rate.append(np.sum(Duration[(Response == 0) & ((Loan != 0))])/np.sum(Duration[(Loan != 0)])*100)

Contact = np.reshape(Contact,(-1,1))
Time_Wasted.append(np.sum(Duration[(Contact == 1)]))
Failure_Rate.append(np.sum(Duration[(Response == 0) & ((Contact == 1))])/np.sum(Duration[(Contact == 1)])*100)

#Saving an output
import csv
with open("output.csv", 'a') as outcsv:
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    writer.writerow(['Time_Wasted', 'Failure_Rate'])
    for i in range(0,8):
        writer.writerow([Time_Wasted[i],Failure_Rate[i]])
