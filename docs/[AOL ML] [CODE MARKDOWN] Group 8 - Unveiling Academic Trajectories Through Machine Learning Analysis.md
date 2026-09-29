```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats.mstats import winsorize

# Oversampling
from imblearn.over_sampling import SMOTE

# Linear Model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Support Vector Machine
from sklearn.svm import SVC

# Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

# Boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# External Gradient Boosting
from lightgbm import LGBMClassifier

# Neural Network
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
```

# 2. Load the dataset


```python
df= pd.read_csv('students_dropout_academic_success.csv')
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marital Status</th>
      <th>Application mode</th>
      <th>Application order</th>
      <th>Course</th>
      <th>Daytime/evening attendance</th>
      <th>Previous qualification</th>
      <th>Previous qualification (grade)</th>
      <th>Nacionality</th>
      <th>Mother's qualification</th>
      <th>Father's qualification</th>
      <th>...</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>17</td>
      <td>5</td>
      <td>171</td>
      <td>1</td>
      <td>1</td>
      <td>122.0</td>
      <td>1</td>
      <td>19</td>
      <td>12</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>15</td>
      <td>1</td>
      <td>9254</td>
      <td>1</td>
      <td>1</td>
      <td>160.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.666667</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>9070</td>
      <td>1</td>
      <td>1</td>
      <td>122.0</td>
      <td>1</td>
      <td>37</td>
      <td>37</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>17</td>
      <td>2</td>
      <td>9773</td>
      <td>1</td>
      <td>1</td>
      <td>122.0</td>
      <td>1</td>
      <td>38</td>
      <td>37</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>12.400000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>Graduate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>39</td>
      <td>1</td>
      <td>8014</td>
      <td>0</td>
      <td>1</td>
      <td>100.0</td>
      <td>1</td>
      <td>37</td>
      <td>38</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>




```python
df.shape
```




    (4424, 37)



We have 4424 data and 37 variablle

# 3. Data Summary


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4424 entries, 0 to 4423
    Data columns (total 37 columns):
     #   Column                                          Non-Null Count  Dtype  
    ---  ------                                          --------------  -----  
     0   Marital Status                                  4424 non-null   int64  
     1   Application mode                                4424 non-null   int64  
     2   Application order                               4424 non-null   int64  
     3   Course                                          4424 non-null   int64  
     4   Daytime/evening attendance                      4424 non-null   int64  
     5   Previous qualification                          4424 non-null   int64  
     6   Previous qualification (grade)                  4424 non-null   float64
     7   Nacionality                                     4424 non-null   int64  
     8   Mother's qualification                          4424 non-null   int64  
     9   Father's qualification                          4424 non-null   int64  
     10  Mother's occupation                             4424 non-null   int64  
     11  Father's occupation                             4424 non-null   int64  
     12  Admission grade                                 4424 non-null   float64
     13  Displaced                                       4424 non-null   int64  
     14  Educational special needs                       4424 non-null   int64  
     15  Debtor                                          4424 non-null   int64  
     16  Tuition fees up to date                         4424 non-null   int64  
     17  Gender                                          4424 non-null   int64  
     18  Scholarship holder                              4424 non-null   int64  
     19  Age at enrollment                               4424 non-null   int64  
     20  International                                   4424 non-null   int64  
     21  Curricular units 1st sem (credited)             4424 non-null   int64  
     22  Curricular units 1st sem (enrolled)             4424 non-null   int64  
     23  Curricular units 1st sem (evaluations)          4424 non-null   int64  
     24  Curricular units 1st sem (approved)             4424 non-null   int64  
     25  Curricular units 1st sem (grade)                4424 non-null   float64
     26  Curricular units 1st sem (without evaluations)  4424 non-null   int64  
     27  Curricular units 2nd sem (credited)             4424 non-null   int64  
     28  Curricular units 2nd sem (enrolled)             4424 non-null   int64  
     29  Curricular units 2nd sem (evaluations)          4424 non-null   int64  
     30  Curricular units 2nd sem (approved)             4424 non-null   int64  
     31  Curricular units 2nd sem (grade)                4424 non-null   float64
     32  Curricular units 2nd sem (without evaluations)  4424 non-null   int64  
     33  Unemployment rate                               4424 non-null   float64
     34  Inflation rate                                  4424 non-null   float64
     35  GDP                                             4424 non-null   float64
     36  target                                          4424 non-null   object 
    dtypes: float64(7), int64(29), object(1)
    memory usage: 1.2+ MB
    


```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Marital Status</th>
      <td>4424.0</td>
      <td>1.178571</td>
      <td>0.605747</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Application mode</th>
      <td>4424.0</td>
      <td>18.669078</td>
      <td>17.484682</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>17.000000</td>
      <td>39.000000</td>
      <td>57.000000</td>
    </tr>
    <tr>
      <th>Application order</th>
      <td>4424.0</td>
      <td>1.727848</td>
      <td>1.313793</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>Course</th>
      <td>4424.0</td>
      <td>8856.642631</td>
      <td>2063.566416</td>
      <td>33.00</td>
      <td>9085.00</td>
      <td>9238.000000</td>
      <td>9556.000000</td>
      <td>9991.000000</td>
    </tr>
    <tr>
      <th>Daytime/evening attendance</th>
      <td>4424.0</td>
      <td>0.890823</td>
      <td>0.311897</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Previous qualification</th>
      <td>4424.0</td>
      <td>4.577758</td>
      <td>10.216592</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>43.000000</td>
    </tr>
    <tr>
      <th>Previous qualification (grade)</th>
      <td>4424.0</td>
      <td>132.613314</td>
      <td>13.188332</td>
      <td>95.00</td>
      <td>125.00</td>
      <td>133.100000</td>
      <td>140.000000</td>
      <td>190.000000</td>
    </tr>
    <tr>
      <th>Nacionality</th>
      <td>4424.0</td>
      <td>1.873192</td>
      <td>6.914514</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>109.000000</td>
    </tr>
    <tr>
      <th>Mother's qualification</th>
      <td>4424.0</td>
      <td>19.561935</td>
      <td>15.603186</td>
      <td>1.00</td>
      <td>2.00</td>
      <td>19.000000</td>
      <td>37.000000</td>
      <td>44.000000</td>
    </tr>
    <tr>
      <th>Father's qualification</th>
      <td>4424.0</td>
      <td>22.275316</td>
      <td>15.343108</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>19.000000</td>
      <td>37.000000</td>
      <td>44.000000</td>
    </tr>
    <tr>
      <th>Mother's occupation</th>
      <td>4424.0</td>
      <td>10.960895</td>
      <td>26.418253</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>5.000000</td>
      <td>9.000000</td>
      <td>194.000000</td>
    </tr>
    <tr>
      <th>Father's occupation</th>
      <td>4424.0</td>
      <td>11.032324</td>
      <td>25.263040</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>7.000000</td>
      <td>9.000000</td>
      <td>195.000000</td>
    </tr>
    <tr>
      <th>Admission grade</th>
      <td>4424.0</td>
      <td>126.978119</td>
      <td>14.482001</td>
      <td>95.00</td>
      <td>117.90</td>
      <td>126.100000</td>
      <td>134.800000</td>
      <td>190.000000</td>
    </tr>
    <tr>
      <th>Displaced</th>
      <td>4424.0</td>
      <td>0.548373</td>
      <td>0.497711</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Educational special needs</th>
      <td>4424.0</td>
      <td>0.011528</td>
      <td>0.106760</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Debtor</th>
      <td>4424.0</td>
      <td>0.113698</td>
      <td>0.317480</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Tuition fees up to date</th>
      <td>4424.0</td>
      <td>0.880651</td>
      <td>0.324235</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>4424.0</td>
      <td>0.351718</td>
      <td>0.477560</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Scholarship holder</th>
      <td>4424.0</td>
      <td>0.248418</td>
      <td>0.432144</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Age at enrollment</th>
      <td>4424.0</td>
      <td>23.265145</td>
      <td>7.587816</td>
      <td>17.00</td>
      <td>19.00</td>
      <td>20.000000</td>
      <td>25.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>International</th>
      <td>4424.0</td>
      <td>0.024864</td>
      <td>0.155729</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Curricular units 1st sem (credited)</th>
      <td>4424.0</td>
      <td>0.709991</td>
      <td>2.360507</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>Curricular units 1st sem (enrolled)</th>
      <td>4424.0</td>
      <td>6.270570</td>
      <td>2.480178</td>
      <td>0.00</td>
      <td>5.00</td>
      <td>6.000000</td>
      <td>7.000000</td>
      <td>26.000000</td>
    </tr>
    <tr>
      <th>Curricular units 1st sem (evaluations)</th>
      <td>4424.0</td>
      <td>8.299051</td>
      <td>4.179106</td>
      <td>0.00</td>
      <td>6.00</td>
      <td>8.000000</td>
      <td>10.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>Curricular units 1st sem (approved)</th>
      <td>4424.0</td>
      <td>4.706600</td>
      <td>3.094238</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>26.000000</td>
    </tr>
    <tr>
      <th>Curricular units 1st sem (grade)</th>
      <td>4424.0</td>
      <td>10.640822</td>
      <td>4.843663</td>
      <td>0.00</td>
      <td>11.00</td>
      <td>12.285714</td>
      <td>13.400000</td>
      <td>18.875000</td>
    </tr>
    <tr>
      <th>Curricular units 1st sem (without evaluations)</th>
      <td>4424.0</td>
      <td>0.137658</td>
      <td>0.690880</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>Curricular units 2nd sem (credited)</th>
      <td>4424.0</td>
      <td>0.541817</td>
      <td>1.918546</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>19.000000</td>
    </tr>
    <tr>
      <th>Curricular units 2nd sem (enrolled)</th>
      <td>4424.0</td>
      <td>6.232143</td>
      <td>2.195951</td>
      <td>0.00</td>
      <td>5.00</td>
      <td>6.000000</td>
      <td>7.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>Curricular units 2nd sem (evaluations)</th>
      <td>4424.0</td>
      <td>8.063291</td>
      <td>3.947951</td>
      <td>0.00</td>
      <td>6.00</td>
      <td>8.000000</td>
      <td>10.000000</td>
      <td>33.000000</td>
    </tr>
    <tr>
      <th>Curricular units 2nd sem (approved)</th>
      <td>4424.0</td>
      <td>4.435805</td>
      <td>3.014764</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>Curricular units 2nd sem (grade)</th>
      <td>4424.0</td>
      <td>10.230206</td>
      <td>5.210808</td>
      <td>0.00</td>
      <td>10.75</td>
      <td>12.200000</td>
      <td>13.333333</td>
      <td>18.571429</td>
    </tr>
    <tr>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <td>4424.0</td>
      <td>0.150316</td>
      <td>0.753774</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>Unemployment rate</th>
      <td>4424.0</td>
      <td>11.566139</td>
      <td>2.663850</td>
      <td>7.60</td>
      <td>9.40</td>
      <td>11.100000</td>
      <td>13.900000</td>
      <td>16.200000</td>
    </tr>
    <tr>
      <th>Inflation rate</th>
      <td>4424.0</td>
      <td>1.228029</td>
      <td>1.382711</td>
      <td>-0.80</td>
      <td>0.30</td>
      <td>1.400000</td>
      <td>2.600000</td>
      <td>3.700000</td>
    </tr>
    <tr>
      <th>GDP</th>
      <td>4424.0</td>
      <td>0.001969</td>
      <td>2.269935</td>
      <td>-4.06</td>
      <td>-1.70</td>
      <td>0.320000</td>
      <td>1.790000</td>
      <td>3.510000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.keys()
```




    Index(['Marital Status', 'Application mode', 'Application order', 'Course',
           'Daytime/evening attendance', 'Previous qualification',
           'Previous qualification (grade)', 'Nacionality',
           'Mother's qualification', 'Father's qualification',
           'Mother's occupation', 'Father's occupation', 'Admission grade',
           'Displaced', 'Educational special needs', 'Debtor',
           'Tuition fees up to date', 'Gender', 'Scholarship holder',
           'Age at enrollment', 'International',
           'Curricular units 1st sem (credited)',
           'Curricular units 1st sem (enrolled)',
           'Curricular units 1st sem (evaluations)',
           'Curricular units 1st sem (approved)',
           'Curricular units 1st sem (grade)',
           'Curricular units 1st sem (without evaluations)',
           'Curricular units 2nd sem (credited)',
           'Curricular units 2nd sem (enrolled)',
           'Curricular units 2nd sem (evaluations)',
           'Curricular units 2nd sem (approved)',
           'Curricular units 2nd sem (grade)',
           'Curricular units 2nd sem (without evaluations)', 'Unemployment rate',
           'Inflation rate', 'GDP', 'target'],
          dtype='object')



# 4. Data Cleaning


```python
print(df['target'].unique())
```

    ['Dropout' 'Graduate' 'Enrolled']
    


```python
# Check Missing Values
df.isna().sum() / len(df) * 100
```




    Marital Status                                    0.0
    Application mode                                  0.0
    Application order                                 0.0
    Course                                            0.0
    Daytime/evening attendance                        0.0
    Previous qualification                            0.0
    Previous qualification (grade)                    0.0
    Nacionality                                       0.0
    Mother's qualification                            0.0
    Father's qualification                            0.0
    Mother's occupation                               0.0
    Father's occupation                               0.0
    Admission grade                                   0.0
    Displaced                                         0.0
    Educational special needs                         0.0
    Debtor                                            0.0
    Tuition fees up to date                           0.0
    Gender                                            0.0
    Scholarship holder                                0.0
    Age at enrollment                                 0.0
    International                                     0.0
    Curricular units 1st sem (credited)               0.0
    Curricular units 1st sem (enrolled)               0.0
    Curricular units 1st sem (evaluations)            0.0
    Curricular units 1st sem (approved)               0.0
    Curricular units 1st sem (grade)                  0.0
    Curricular units 1st sem (without evaluations)    0.0
    Curricular units 2nd sem (credited)               0.0
    Curricular units 2nd sem (enrolled)               0.0
    Curricular units 2nd sem (evaluations)            0.0
    Curricular units 2nd sem (approved)               0.0
    Curricular units 2nd sem (grade)                  0.0
    Curricular units 2nd sem (without evaluations)    0.0
    Unemployment rate                                 0.0
    Inflation rate                                    0.0
    GDP                                               0.0
    target                                            0.0
    dtype: float64



There's no missing value in this dataset


```python
# Check duplicate data
df.duplicated().sum()
```




    np.int64(0)



There's no duplicate value in this dataset

# 5. Data Visualisation


```python
num_cols = df.select_dtypes(include = ['int64','float64']).columns
cat_cols = df.select_dtypes(exclude = ['int64','float64']).columns
```


```python
num_rows = (len(num_cols) + 1) // 2

f, ax = plt.subplots(9,4, figsize=(25,2 * num_rows))
ax = ax.flatten()

for index, cols in enumerate(num_cols):
    sns.histplot(data = df, x= cols, ax = ax[index],kde=True, bins=30)
    ax[index].set_title(cols, size = 20)
    
plt.tight_layout()
plt.show()
```


    
![png](output_17_0.png)
    



```python
num_rows = (len(num_cols) + 1) // 2

f, ax = plt.subplots(9,4, figsize=(25,2 * num_rows))
ax = ax.flatten()

for index, cols in enumerate(num_cols):
    sns.boxplot(df, x= cols , ax= ax[index])
    ax[index].set_title(cols, size=20)
    
plt.tight_layout()
plt.show()
```


    
![png](output_18_0.png)
    



```python
plt.figure(figsize=(10, 6))
sns.countplot(df, x = 'target')
plt.title('Target Variables CountPlot')
plt.show()
```


    
![png](output_19_0.png)
    



```python
labels = ['Graduate','Dropout', 'Enrolled']

plt.figure(figsize=(10, 6))
plt.pie(df['target'].value_counts() , labels = labels, autopct='%.1f%%' )
plt.title('Target Variables PiePlot')
plt.show()
```


    
![png](output_20_0.png)
    



```python
plt.figure(figsize=(20, 10))
sns.heatmap(df[num_cols].corr(),annot=True, cmap='Reds',fmt= '.2f')
plt.show()
```


    
![png](output_21_0.png)
    


# 6. Feature Engineering

## Outlier Handling


```python
from scipy.stats.mstats import winsorize

df_transformed = df.copy()

# --- Log Transform (kolom skewed dan selalu positif) ---
log_transform_cols = [
    'Application order',
    'Age at enrollment',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 2nd sem (enrolled)'
]

for col in log_transform_cols:
    df_transformed[col] = np.log1p(df_transformed[col])  # log(1+x)

# --- Winsorize (batasi outlier ekstrem) ---
winsorize_cols = [
    'Admission grade',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (grade)',
    'Mother\'s occupation',
    'Father\'s occupation'
]

for col in winsorize_cols:
    df_transformed[col] = winsorize(df_transformed[col], limits=[0.01, 0.01])

# --- Cek hasil transformasi ---
df_transformed[log_transform_cols + winsorize_cols].describe()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Application order</th>
      <th>Age at enrollment</th>
      <th>Curricular units 1st sem (evaluations)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 1st sem (enrolled)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Admission grade</th>
      <th>Curricular units 1st sem (grade)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Mother's occupation</th>
      <th>Father's occupation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.920511</td>
      <td>3.151732</td>
      <td>2.075025</td>
      <td>2.041237</td>
      <td>1.909321</td>
      <td>1.910361</td>
      <td>126.927080</td>
      <td>10.634238</td>
      <td>10.224007</td>
      <td>10.950497</td>
      <td>10.888788</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.375216</td>
      <td>0.256310</td>
      <td>0.676115</td>
      <td>0.702082</td>
      <td>0.452198</td>
      <td>0.441691</td>
      <td>14.188349</td>
      <td>4.835557</td>
      <td>5.203243</td>
      <td>26.346764</td>
      <td>24.278014</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>2.890372</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>99.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.693147</td>
      <td>2.995732</td>
      <td>1.945910</td>
      <td>1.945910</td>
      <td>1.791759</td>
      <td>1.791759</td>
      <td>117.900000</td>
      <td>11.000000</td>
      <td>10.750000</td>
      <td>4.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.693147</td>
      <td>3.044522</td>
      <td>2.197225</td>
      <td>2.197225</td>
      <td>1.945910</td>
      <td>1.945910</td>
      <td>126.100000</td>
      <td>12.285714</td>
      <td>12.200000</td>
      <td>5.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.098612</td>
      <td>3.258097</td>
      <td>2.397895</td>
      <td>2.397895</td>
      <td>2.079442</td>
      <td>2.079442</td>
      <td>134.800000</td>
      <td>13.400000</td>
      <td>13.333333</td>
      <td>9.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.302585</td>
      <td>4.262680</td>
      <td>3.828641</td>
      <td>3.526361</td>
      <td>3.295837</td>
      <td>3.178054</td>
      <td>166.900000</td>
      <td>16.000000</td>
      <td>16.000000</td>
      <td>191.000000</td>
      <td>171.000000</td>
    </tr>
  </tbody>
</table>
</div>



Dalam preprocessing data, saya menggunakan log transform dan winsorize untuk menangani distribusi yang tidak normal dan outlier.

Log transform digunakan pada fitur-fitur yang memiliki sebaran skewed (miring ke kanan) dan hanya berisi nilai positif. Tujuannya untuk menormalkan distribusi, mengurangi pengaruh nilai ekstrem, dan membantu model yang sensitif terhadap skala.

Winsorize digunakan untuk membatasi outlier ekstrem dengan menggantinya pada batas persentil tertentu (misalnya 1% dan 99%), agar data tidak terlalu terpengaruh oleh nilai yang tidak representatif tanpa harus menghapusnya.

Kedua teknik ini dipilih karena lebih efektif dibanding Z-score atau IQR dalam kasus data yang tidak normal dan mengandung banyak outlier, seperti pada dataset ini yang memiliki target multiclass.

## Encoding


```python
from sklearn.preprocessing import OneHotEncoder

onehot_cols = [
    'Marital Status', 'Application mode', 'Course', 'Daytime/evening attendance',
    'Nacionality', "Mother's occupation", "Father's occupation",
    'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International',"Mother's qualification",
    "Father's qualification",'Previous qualification'
]

df_transformed = pd.get_dummies(df_transformed, columns=onehot_cols, drop_first=True)
```


```python
df_transformed.head(5).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Application order</th>
      <td>1.791759</td>
      <td>0.693147</td>
      <td>1.791759</td>
      <td>1.098612</td>
      <td>0.693147</td>
    </tr>
    <tr>
      <th>Previous qualification (grade)</th>
      <td>122.0</td>
      <td>160.0</td>
      <td>122.0</td>
      <td>122.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Admission grade</th>
      <td>127.3</td>
      <td>142.5</td>
      <td>124.8</td>
      <td>119.6</td>
      <td>141.5</td>
    </tr>
    <tr>
      <th>Age at enrollment</th>
      <td>3.044522</td>
      <td>2.995732</td>
      <td>2.995732</td>
      <td>3.044522</td>
      <td>3.828641</td>
    </tr>
    <tr>
      <th>Curricular units 1st sem (credited)</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Previous qualification_38</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Previous qualification_39</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Previous qualification_40</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Previous qualification_42</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>Previous qualification_43</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>226 rows × 5 columns</p>
</div>




```python
df_transformed.shape
```




    (4424, 226)



One-Hot Encoding digunakan untuk mengubah fitur kategorik nominal menjadi representasi numerik yang dapat dibaca oleh model machine learning. Teknik ini mencegah model mengasumsikan adanya urutan atau hubungan matematis antar kategori, karena setiap kategori diubah menjadi kolom biner (0 atau 1).

## Scaling


```python
numeric_cols = df_transformed.select_dtypes(include='number').columns.tolist()

numeric_cols = [col for col in numeric_cols if df_transformed[col].nunique() > 2]

# Inisialisasi dan scaling
scaler = StandardScaler()
df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])

# Cek hasil
df_transformed[numeric_cols].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Application order</th>
      <th>Previous qualification (grade)</th>
      <th>Admission grade</th>
      <th>Age at enrollment</th>
      <th>Curricular units 1st sem (credited)</th>
      <th>Curricular units 1st sem (enrolled)</th>
      <th>Curricular units 1st sem (evaluations)</th>
      <th>Curricular units 1st sem (approved)</th>
      <th>Curricular units 1st sem (grade)</th>
      <th>Curricular units 1st sem (without evaluations)</th>
      <th>Curricular units 2nd sem (credited)</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
      <td>4.424000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-1.188521e-16</td>
      <td>-3.589654e-16</td>
      <td>-7.195369e-16</td>
      <td>1.856662e-15</td>
      <td>3.212219e-17</td>
      <td>5.187733e-16</td>
      <td>-9.476045e-17</td>
      <td>3.051608e-17</td>
      <td>-1.509743e-16</td>
      <td>9.636656e-18</td>
      <td>-3.212219e-18</td>
      <td>5.067275e-16</td>
      <td>-1.509743e-16</td>
      <td>-1.509743e-16</td>
      <td>1.220643e-16</td>
      <td>3.212219e-18</td>
      <td>-5.460771e-17</td>
      <td>1.445498e-16</td>
      <td>2.569775e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
      <td>1.000113e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.453564e+00</td>
      <td>-2.852337e+00</td>
      <td>-1.968533e+00</td>
      <td>-1.019819e+00</td>
      <td>-3.008131e-01</td>
      <td>-4.222784e+00</td>
      <td>-3.069388e+00</td>
      <td>-1.521257e+00</td>
      <td>-2.199424e+00</td>
      <td>-1.992730e-01</td>
      <td>-2.824423e-01</td>
      <td>-4.325593e+00</td>
      <td>-2.907733e+00</td>
      <td>-1.471527e+00</td>
      <td>-1.965152e+00</td>
      <td>-1.994410e-01</td>
      <td>-1.489043e+00</td>
      <td>-1.466871e+00</td>
      <td>-1.789667e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-6.060248e-01</td>
      <td>-5.773417e-01</td>
      <td>-6.363038e-01</td>
      <td>-6.087061e-01</td>
      <td>-3.008131e-01</td>
      <td>-2.600061e-01</td>
      <td>-1.909879e-01</td>
      <td>-5.516038e-01</td>
      <td>7.564871e-02</td>
      <td>-1.992730e-01</td>
      <td>-2.824423e-01</td>
      <td>-2.685470e-01</td>
      <td>-1.357924e-01</td>
      <td>-8.080500e-01</td>
      <td>1.011008e-01</td>
      <td>-1.994410e-01</td>
      <td>-8.132529e-01</td>
      <td>-6.712423e-01</td>
      <td>-7.498721e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-6.060248e-01</td>
      <td>3.690696e-02</td>
      <td>-5.829946e-02</td>
      <td>-4.183285e-01</td>
      <td>-3.008131e-01</td>
      <td>8.092410e-02</td>
      <td>1.807577e-01</td>
      <td>9.483201e-02</td>
      <td>3.415663e-01</td>
      <td>-1.992730e-01</td>
      <td>-2.824423e-01</td>
      <td>8.049339e-02</td>
      <td>2.222038e-01</td>
      <td>1.871653e-01</td>
      <td>3.798047e-01</td>
      <td>-1.994410e-01</td>
      <td>-1.750068e-01</td>
      <td>1.243865e-01</td>
      <td>1.401217e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.747164e-01</td>
      <td>5.601559e-01</td>
      <td>5.549490e-01</td>
      <td>4.150305e-01</td>
      <td>-3.008131e-01</td>
      <td>3.762512e-01</td>
      <td>4.775908e-01</td>
      <td>4.180499e-01</td>
      <td>5.720282e-01</td>
      <td>-1.992730e-01</td>
      <td>-2.824423e-01</td>
      <td>3.828459e-01</td>
      <td>5.080584e-01</td>
      <td>5.189037e-01</td>
      <td>5.976422e-01</td>
      <td>-1.994410e-01</td>
      <td>8.762221e-01</td>
      <td>9.923451e-01</td>
      <td>7.877904e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.683829e+00</td>
      <td>4.351815e+00</td>
      <td>2.817625e+00</td>
      <td>4.334881e+00</td>
      <td>8.172902e+00</td>
      <td>3.066514e+00</td>
      <td>2.593958e+00</td>
      <td>6.882407e+00</td>
      <td>1.109773e+00</td>
      <td>1.717184e+01</td>
      <td>9.622009e+00</td>
      <td>2.870413e+00</td>
      <td>2.115552e+00</td>
      <td>5.163242e+00</td>
      <td>1.110201e+00</td>
      <td>1.572225e+01</td>
      <td>1.739731e+00</td>
      <td>1.787974e+00</td>
      <td>1.545607e+00</td>
    </tr>
  </tbody>
</table>
</div>



Saya menggunakan StandardScaler untuk menormalkan fitur numerik agar memiliki mean 0 dan standar deviasi 1. Ini penting agar algoritma seperti KNN, SVM, dan Logistic Regression bisa bekerja optimal, karena mereka sensitif terhadap skala antar fitur.


```python
df_transformed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Application order</th>
      <th>Previous qualification (grade)</th>
      <th>Admission grade</th>
      <th>Age at enrollment</th>
      <th>Curricular units 1st sem (credited)</th>
      <th>Curricular units 1st sem (enrolled)</th>
      <th>Curricular units 1st sem (evaluations)</th>
      <th>Curricular units 1st sem (approved)</th>
      <th>Curricular units 1st sem (grade)</th>
      <th>Curricular units 1st sem (without evaluations)</th>
      <th>...</th>
      <th>Previous qualification_10</th>
      <th>Previous qualification_12</th>
      <th>Previous qualification_14</th>
      <th>Previous qualification_15</th>
      <th>Previous qualification_19</th>
      <th>Previous qualification_38</th>
      <th>Previous qualification_39</th>
      <th>Previous qualification_40</th>
      <th>Previous qualification_42</th>
      <th>Previous qualification_43</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.322256</td>
      <td>-0.804841</td>
      <td>0.026287</td>
      <td>-0.418328</td>
      <td>-0.300813</td>
      <td>-4.222784</td>
      <td>-3.069388</td>
      <td>-1.521257</td>
      <td>-2.199424</td>
      <td>-0.199273</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.606025</td>
      <td>2.076819</td>
      <td>1.097709</td>
      <td>-0.608706</td>
      <td>-0.300813</td>
      <td>0.080924</td>
      <td>-0.190988</td>
      <td>0.418050</td>
      <td>0.696123</td>
      <td>-0.199273</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.322256</td>
      <td>-0.804841</td>
      <td>-0.149934</td>
      <td>-0.608706</td>
      <td>-0.300813</td>
      <td>0.080924</td>
      <td>-3.069388</td>
      <td>-1.521257</td>
      <td>-2.199424</td>
      <td>-0.199273</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.474716</td>
      <td>-0.804841</td>
      <td>-0.516474</td>
      <td>-0.418328</td>
      <td>-0.300813</td>
      <td>0.080924</td>
      <td>0.180758</td>
      <td>0.418050</td>
      <td>0.577937</td>
      <td>-0.199273</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.606025</td>
      <td>-2.473171</td>
      <td>1.027221</td>
      <td>2.641277</td>
      <td>-0.300813</td>
      <td>0.080924</td>
      <td>0.336607</td>
      <td>0.094832</td>
      <td>0.351415</td>
      <td>-0.199273</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 226 columns</p>
</div>



# 7. Mechine Learning Model 

## Feature Selection


```python
y = df_transformed['target']
X = df_transformed.drop('target', axis=1)
```

## Data Splitting


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 100)

print(f'Total: {len(X)}')
print(f'Total train sample: {len(X_train)}')
print(f'Total test sample: {len(X_test)}')
```

    Total: 4424
    Total train sample: 3539
    Total test sample: 885
    

## Oversampling


```python
sns.countplot(data=df, x= 'target')
plt.show()
```


    
![png](output_41_0.png)
    



```python
df['target'].value_counts(normalize=True) * 100
```




    target
    Graduate    49.932188
    Dropout     32.120253
    Enrolled    17.947559
    Name: proportion, dtype: float64



imblance data, so kita perlu oversampling menggunakan SMOTE

SMOTE digunakan karena data target tidak seimbang — mayoritas kelas adalah Graduate (49.9%), sedangkan Enrolled hanya 17.9%. Untuk mencegah model bias terhadap kelas mayoritas, SMOTE menambahkan data sintetis ke kelas minoritas agar model bisa belajar secara adil dari semua kelas.


```python
# Oversampling menggunakan SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(y_train_resampled.value_counts())
```

    target
    Dropout     1752
    Graduate    1752
    Enrolled    1752
    Name: count, dtype: int64
    


```python
labels = ['Graduate','Dropout', 'Enrolled']

plt.figure(figsize=(10, 6))
plt.pie(y_train_resampled.value_counts() , labels = labels, autopct='%.1f%%' )
plt.title('Target Variables PiePlot')
plt.show()
```


    
![png](output_46_0.png)
    


## Model Training


```python
# 1. Logistic Regression
model_lr = LogisticRegression(random_state=100)
model_lr.fit(X_train_resampled, y_train_resampled)
y_pred_lr = model_lr.predict(X_test)
```

    C:\Users\Mumtaz\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\linear_model\_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    


```python
# 2. SGD Classifier
model_sgd = SGDClassifier(random_state=100)
model_sgd.fit(X_train_resampled, y_train_resampled)
y_pred_sgd = model_sgd.predict(X_test)
```


```python
# 3. Linear Discriminant Analysis
model_lda = LinearDiscriminantAnalysis()
model_lda.fit(X_train_resampled, y_train_resampled)
y_pred_lda = model_lda.predict(X_test)
```


```python
# 4. Quadratic Discriminant Analysis
model_qda = QuadraticDiscriminantAnalysis()
model_qda.fit(X_train_resampled, y_train_resampled)
y_pred_qda = model_qda.predict(X_test)
```

    C:\Users\Mumtaz\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\discriminant_analysis.py:1024: LinAlgWarning: The covariance matrix of class 0 is not full rank. Increasing the value of parameter `reg_param` might help reducing the collinearity.
      warnings.warn(
    C:\Users\Mumtaz\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\discriminant_analysis.py:1024: LinAlgWarning: The covariance matrix of class 1 is not full rank. Increasing the value of parameter `reg_param` might help reducing the collinearity.
      warnings.warn(
    C:\Users\Mumtaz\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\discriminant_analysis.py:1024: LinAlgWarning: The covariance matrix of class 2 is not full rank. Increasing the value of parameter `reg_param` might help reducing the collinearity.
      warnings.warn(
    


```python
# 5. Support Vector Machine
model_svc = SVC(random_state=100)
model_svc.fit(X_train_resampled, y_train_resampled)
y_pred_svc = model_svc.predict(X_test)
```


```python
# 6. Decision Tree
model_dt = DecisionTreeClassifier(random_state=100)
model_dt.fit(X_train_resampled, y_train_resampled)
y_pred_dt = model_dt.predict(X_test)

```


```python
# 7. Extra Trees
model_et = ExtraTreesClassifier(random_state=100)
model_et.fit(X_train_resampled, y_train_resampled)
y_pred_et = model_et.predict(X_test)
```


```python
# 8. Random Forest
model_rf = RandomForestClassifier(random_state=100)
model_rf.fit(X_train_resampled, y_train_resampled)
y_pred_rf = model_rf.predict(X_test)
```


```python
# 9. Gradient Boosting
model_gb = GradientBoostingClassifier(random_state=100)
model_gb.fit(X_train_resampled, y_train_resampled)
y_pred_gb = model_gb.predict(X_test)
```


```python
# 10. Hist Gradient Boosting
model_hgb = HistGradientBoostingClassifier(random_state=100)
model_hgb.fit(X_train_resampled, y_train_resampled)
y_pred_hgb = model_hgb.predict(X_test)
```


```python
# 11. AdaBoost
model_ada = AdaBoostClassifier(random_state=100)
model_ada.fit(X_train_resampled, y_train_resampled)
y_pred_ada = model_ada.predict(X_test)
```


```python
# 12. Gaussian Naive Bayes
model_nb = GaussianNB()
model_nb.fit(X_train_resampled, y_train_resampled)
y_pred_nb = model_nb.predict(X_test)
```


```python
# 13. K-Nearest Neighbors
model_knn = KNeighborsClassifier()
model_knn.fit(X_train_resampled, y_train_resampled)
y_pred_knn = model_knn.predict(X_test)
```


```python
# 14. LightGBM
model_lgbm = LGBMClassifier(random_state=42)
model_lgbm.fit(X_train_resampled, y_train_resampled)
y_pred_lgbm = model_lgbm.predict(X_test)
```

    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.001522 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 4219
    [LightGBM] [Info] Number of data points in the train set: 5256, number of used features: 114
    [LightGBM] [Info] Start training from score -1.098612
    [LightGBM] [Info] Start training from score -1.098612
    [LightGBM] [Info] Start training from score -1.098612
    


```python
# 15. MLP Classifier
model_mlp = MLPClassifier(random_state=42)
model_mlp.fit(X_train_resampled, y_train_resampled)
y_pred_mlp = model_mlp.predict(X_test)
```

    C:\Users\Mumtaz\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\sklearn\neural_network\_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      warnings.warn(
    

# 8. Model Evaluation

## Before Hypertunning and choose best model


```python
# ROC AUC Comparison
plt.figure(figsize=(12, 8))

models = {
    "Logistic Regression": model_lr,
    "SGD": model_sgd,
    "LDA": model_lda,
    "QDA": model_qda,
    "SVC": model_svc,
    "Decision Tree": model_dt,
    "Extra Trees": model_et,
    "Random Forest": model_rf,
    "GB": model_gb,
    "HistGB": model_hgb,
    "AdaBoost": model_ada,
    "Gaussian NB": model_nb,
    "KNN": model_knn,
    "LightGBM": model_lgbm,
    "MLP": model_mlp
}
```


    <Figure size 1200x800 with 0 Axes>



```python
# Simpan confusion matrices dan metrics
all_cms = []
metrics = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": [],
    "Correct": [],
    "Total": [],
    "Misclassified": []
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    all_cms.append(cm)

    correct = cm.trace()  # jumlah nilai diagonal (prediksi benar)
    total = cm.sum()
    misclassified = total - correct

    metrics["Model"].append(name)
    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
    metrics["Precision"].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    metrics["Recall"].append(recall_score(y_test, y_pred, average='weighted'))
    metrics["F1 Score"].append(f1_score(y_test, y_pred, average='weighted'))
    metrics["Correct"].append(correct)
    metrics["Total"].append(total)
    metrics["Misclassified"].append(misclassified)

# Buat DataFrame
df_metrics = pd.DataFrame(metrics)

# Urutkan dari model terbaik (akurasi tertinggi)
df_sorted = df_metrics.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

# Tambahkan kolom peringkat
df_sorted.insert(0, 'Rank', df_sorted.index + 1)

# Tampilkan tabel
display(df_sorted[["Rank", "Model", "Accuracy", "Correct", "Misclassified", "Precision", "Recall", "F1 Score"]])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Correct</th>
      <th>Misclassified</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>HistGB</td>
      <td>0.794350</td>
      <td>703</td>
      <td>182</td>
      <td>0.783439</td>
      <td>0.794350</td>
      <td>0.786497</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>LightGBM</td>
      <td>0.794350</td>
      <td>703</td>
      <td>182</td>
      <td>0.784475</td>
      <td>0.794350</td>
      <td>0.787278</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Random Forest</td>
      <td>0.788701</td>
      <td>698</td>
      <td>187</td>
      <td>0.783818</td>
      <td>0.788701</td>
      <td>0.783671</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>SVC</td>
      <td>0.779661</td>
      <td>690</td>
      <td>195</td>
      <td>0.776769</td>
      <td>0.779661</td>
      <td>0.775836</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Logistic Regression</td>
      <td>0.779661</td>
      <td>690</td>
      <td>195</td>
      <td>0.775014</td>
      <td>0.779661</td>
      <td>0.776310</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Extra Trees</td>
      <td>0.775141</td>
      <td>686</td>
      <td>199</td>
      <td>0.764491</td>
      <td>0.775141</td>
      <td>0.766961</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>SGD</td>
      <td>0.775141</td>
      <td>686</td>
      <td>199</td>
      <td>0.770537</td>
      <td>0.775141</td>
      <td>0.769049</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>GB</td>
      <td>0.775141</td>
      <td>686</td>
      <td>199</td>
      <td>0.779218</td>
      <td>0.775141</td>
      <td>0.776454</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>LDA</td>
      <td>0.763842</td>
      <td>676</td>
      <td>209</td>
      <td>0.755990</td>
      <td>0.763842</td>
      <td>0.755584</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>AdaBoost</td>
      <td>0.728814</td>
      <td>645</td>
      <td>240</td>
      <td>0.763110</td>
      <td>0.728814</td>
      <td>0.739713</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>MLP</td>
      <td>0.719774</td>
      <td>637</td>
      <td>248</td>
      <td>0.712294</td>
      <td>0.719774</td>
      <td>0.715513</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>KNN</td>
      <td>0.691525</td>
      <td>612</td>
      <td>273</td>
      <td>0.694544</td>
      <td>0.691525</td>
      <td>0.691783</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>Decision Tree</td>
      <td>0.681356</td>
      <td>603</td>
      <td>282</td>
      <td>0.690740</td>
      <td>0.681356</td>
      <td>0.685637</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Gaussian NB</td>
      <td>0.229379</td>
      <td>203</td>
      <td>682</td>
      <td>0.471096</td>
      <td>0.229379</td>
      <td>0.149857</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>QDA</td>
      <td>0.225989</td>
      <td>200</td>
      <td>685</td>
      <td>0.490455</td>
      <td>0.225989</td>
      <td>0.145788</td>
    </tr>
  </tbody>
</table>
</div>


Dari hasil diatas kita mau ambil 5 Model Terbaik Berdasarkan Hasil Evaluasi,


Alasan Memilih 5 Model Terbaik Berdasarkan Hasil Evaluasi

HistGradientBoosting dipilih karena model ini sangat cocok untuk data yang sudah mengalami banyak feature engineering, seperti one-hot encoding dan log transform. Selain cepat dan efisien, model ini juga mampu menangani data dalam jumlah besar serta bekerja baik pada fitur numerik dan kategorikal. Hasil akurasinya juga menjadi yang tertinggi dari seluruh model, menunjukkan kemampuannya dalam memahami pola yang kompleks dalam dataset ini.

LightGBM menjadi pilihan karena ia merupakan boosting model yang sangat efisien untuk data berdimensi tinggi—cocok dengan kondisi dataset yang sudah diperluas melalui encoding. LightGBM juga dikenal sangat baik dalam menangani data imbalance, yang relevan dengan kondisi target variabelmu. Selain itu, performanya setara dengan HistGB, menjadikannya alternatif kuat dengan kecepatan pelatihan yang tinggi.

Random Forest dipilih karena model ini stabil, tidak terlalu sensitif terhadap outlier, dan dapat memberikan hasil yang cukup baik tanpa perlu banyak tuning. Ini cocok untuk kamu yang ingin model yang kuat dan mudah diterapkan sebagai baseline, terutama setelah transformasi seperti winsorizing dan scaling.

SVC (Support Vector Classifier) menjadi kandidat karena model ini unggul dalam membentuk boundary yang jelas antara kelas, terutama jika kelas-kelas memiliki batasan yang tidak linier. Karena datamu telah diskalakan dengan StandardScaler, SVC dapat bekerja optimal. Meski butuh waktu pelatihan lebih lama, performanya menunjukkan model ini mampu beradaptasi dengan kompleksitas fitur yang dihasilkan.

Logistic Regression tetap dipertahankan karena model ini sederhana, mudah diinterpretasikan, dan menjadi acuan yang baik dalam membandingkan model lain. Dengan data yang telah distandarisasi dan ditransformasi, Logistic Regression masih mampu memberikan akurasi dan F1-score yang cukup tinggi, membuktikan bahwa bahkan model yang simpel bisa bersaing saat datanya diolah dengan baik.

## Hypertunning Using RandomSearch

Kenapa pakai RandomizedSearchCV?
Lebih cepat & efisien – tidak mencoba semua kombinasi seperti GridSearchCV.

Cocok untuk banyak parameter – bisa menjelajahi ruang parameter besar tanpa butuh waktu lama.

Bisa sampling dari distribusi – fleksibel untuk nilai continuous seperti learning rate.

Hasil cukup optimal – sering menemukan kombinasi bagus dengan lebih sedikit percobaan.

Hemat resource – cocok kalau pakai laptop atau waktu training terbatas.


```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier

print("Tuning HistGradientBoosting...")
hgb_params = {
    'max_iter': [100, 200, 300, 400],
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'max_depth': [None, 5, 10, 15],
    'l2_regularization': [0, 0.1, 0.3, 0.5, 1.0]
}

rsearch_hgb = RandomizedSearchCV(
    HistGradientBoostingClassifier(random_state=100),
    hgb_params,
    n_iter=30,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=100
)
rsearch_hgb.fit(X_train_resampled, y_train_resampled)
print(f"Best HistGB: {rsearch_hgb.best_params_}, F1: {rsearch_hgb.best_score_}")
```

    Tuning HistGradientBoosting...
    Best HistGB: {'max_iter': 400, 'max_depth': None, 'learning_rate': np.float64(0.20333333333333334), 'l2_regularization': 0.1}, F1: 0.8274310373417532
    


```python
print("\nTuning LightGBM...")
lgbm_params = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'num_leaves': [15, 31, 50, 75, 100],
    'reg_alpha': [0, 0.1, 0.3, 0.5, 1.0],
    'reg_lambda': [0, 0.1, 0.3, 0.5, 1.0]
}
rsearch_lgbm = RandomizedSearchCV(
    LGBMClassifier(random_state=100),
    lgbm_params,
    n_iter=30,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=100
)
rsearch_lgbm.fit(X_train_resampled, y_train_resampled)
print(f"Best LGBM: {rsearch_lgbm.best_params_}, F1: {rsearch_lgbm.best_score_}")
```

    
    Tuning LightGBM...
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.002511 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 4219
    [LightGBM] [Info] Number of data points in the train set: 5256, number of used features: 114
    [LightGBM] [Info] Start training from score -1.098612
    [LightGBM] [Info] Start training from score -1.098612
    [LightGBM] [Info] Start training from score -1.098612
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    Best LGBM: {'reg_lambda': 0, 'reg_alpha': 0, 'num_leaves': 50, 'n_estimators': 300, 'learning_rate': np.float64(0.20333333333333334)}, F1: 0.8302904158853636
    


```python
print("\nTuning Random Forest...")
rf_params = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3, 4]
}
rsearch_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=100),
    rf_params,
    n_iter=30,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=100
)
rsearch_rf.fit(X_train_resampled, y_train_resampled)
print(f"Best RF: {rsearch_rf.best_params_}, F1: {rsearch_rf.best_score_}")
```

    
    Tuning Random Forest...
    Best RF: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': None}, F1: 0.8489052120930051
    


```python
print("\nTuning Support Vector Classifier...")
svc_params = {
    'C': np.logspace(-2, 2, 10),
    'gamma': ['scale', 'auto'] + list(np.logspace(-2, 1, 4)),
    'kernel': ['rbf', 'linear']
}
rsearch_svc = RandomizedSearchCV(
    SVC(random_state=100),
    svc_params,
    n_iter=20,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=100
)
rsearch_svc.fit(X_train_resampled, y_train_resampled)
print(f"Best SVC: {rsearch_svc.best_params_}, F1: {rsearch_svc.best_score_}")
```

    
    Tuning Support Vector Classifier...
    Best SVC: {'kernel': 'rbf', 'gamma': np.float64(0.1), 'C': np.float64(0.5994842503189409)}, F1: 0.8239766869483249
    


```python
print("\nTuning Logistic Regression...")
lr_params = {
    'C': np.logspace(-3, 2, 10),
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs']
}
rsearch_lr = RandomizedSearchCV(
    LogisticRegression(random_state=100, max_iter=2000),
    lr_params,
    n_iter=15,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=100
)
rsearch_lr.fit(X_train_resampled, y_train_resampled)
print(f"Best LR: {rsearch_lr.best_params_}, F1: {rsearch_lr.best_score_}")

```

    
    Tuning Logistic Regression...
    Best LR: {'solver': 'liblinear', 'penalty': 'l2', 'C': np.float64(0.046415888336127795)}, F1: 0.7470242406719972
    

## FINAL MODEL EVALUATION AFTER TUNNING


```python
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Gunakan estimator terbaik dari hasil RandomizedSearchCV
tuned_models = {
    "HistGradientBoosting (Tuned)": rsearch_hgb.best_estimator_,
    "LightGBM (Tuned)": rsearch_lgbm.best_estimator_,
    "Random Forest (Tuned)": rsearch_rf.best_estimator_,
    "SVC (Tuned)": rsearch_svc.best_estimator_,
    "Logistic Regression (Tuned)": rsearch_lr.best_estimator_
}

# Siapkan dictionary untuk menyimpan metrik
tuned_metrics = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

# Evaluasi semua model
for name, model in tuned_models.items():
    y_pred = model.predict(X_test)

    tuned_metrics["Model"].append(name)
    tuned_metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
    tuned_metrics["Precision"].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    tuned_metrics["Recall"].append(recall_score(y_test, y_pred, average='weighted'))
    tuned_metrics["F1 Score"].append(f1_score(y_test, y_pred, average='weighted'))

# Buat DataFrame dan urutkan berdasarkan akurasi
df_tuned_metrics = pd.DataFrame(tuned_metrics)
df_tuned_sorted = df_tuned_metrics.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
df_tuned_sorted.insert(0, 'Rank', df_tuned_sorted.index + 1)

# Tampilkan hasil akhir
display(df_tuned_sorted[["Rank", "Model", "Accuracy", "Precision", "Recall", "F1 Score"]])

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>HistGradientBoosting (Tuned)</td>
      <td>0.955932</td>
      <td>0.955959</td>
      <td>0.955932</td>
      <td>0.955537</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>LightGBM (Tuned)</td>
      <td>0.955932</td>
      <td>0.955927</td>
      <td>0.955932</td>
      <td>0.955553</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Random Forest (Tuned)</td>
      <td>0.954802</td>
      <td>0.955424</td>
      <td>0.954802</td>
      <td>0.954191</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>SVC (Tuned)</td>
      <td>0.850847</td>
      <td>0.848505</td>
      <td>0.850847</td>
      <td>0.842047</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Logistic Regression (Tuned)</td>
      <td>0.794350</td>
      <td>0.781923</td>
      <td>0.794350</td>
      <td>0.772743</td>
    </tr>
  </tbody>
</table>
</div>



```python
tuned_models = {
    "HistGradientBoosting (Tuned)": rsearch_hgb.best_estimator_,
    "LightGBM (Tuned)": rsearch_lgbm.best_estimator_,
    "Random Forest (Tuned)": rsearch_rf.best_estimator_,
    "SVC (Tuned)": rsearch_svc.best_estimator_,
    "Logistic Regression (Tuned)": rsearch_lr.best_estimator_
}

for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name}:\n{classification_report(y_test, y_pred)}")

```

    
    HistGradientBoosting (Tuned):
                  precision    recall  f1-score   support
    
         Dropout       0.97      0.95      0.96       264
        Enrolled       0.95      0.88      0.91       164
        Graduate       0.95      0.98      0.97       457
    
        accuracy                           0.96       885
       macro avg       0.96      0.94      0.95       885
    weighted avg       0.96      0.96      0.96       885
    
    
    LightGBM (Tuned):
                  precision    recall  f1-score   support
    
         Dropout       0.97      0.95      0.96       264
        Enrolled       0.94      0.88      0.91       164
        Graduate       0.95      0.98      0.97       457
    
        accuracy                           0.96       885
       macro avg       0.96      0.94      0.95       885
    weighted avg       0.96      0.96      0.96       885
    
    
    Random Forest (Tuned):
                  precision    recall  f1-score   support
    
         Dropout       0.97      0.96      0.97       264
        Enrolled       0.97      0.85      0.91       164
        Graduate       0.94      0.99      0.96       457
    
        accuracy                           0.95       885
       macro avg       0.96      0.93      0.95       885
    weighted avg       0.96      0.95      0.95       885
    
    
    SVC (Tuned):
                  precision    recall  f1-score   support
    
         Dropout       0.90      0.84      0.87       264
        Enrolled       0.78      0.52      0.63       164
        Graduate       0.84      0.98      0.90       457
    
        accuracy                           0.85       885
       macro avg       0.84      0.78      0.80       885
    weighted avg       0.85      0.85      0.84       885
    
    
    Logistic Regression (Tuned):
                  precision    recall  f1-score   support
    
         Dropout       0.83      0.78      0.80       264
        Enrolled       0.66      0.32      0.43       164
        Graduate       0.80      0.97      0.88       457
    
        accuracy                           0.79       885
       macro avg       0.76      0.69      0.70       885
    weighted avg       0.78      0.79      0.77       885
    
    


```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 8))
cv = StratifiedKFold(n_splits=5)
n_classes = len(np.unique(y))  # jumlah kelas

for name, model in tuned_models.items():
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for train_index, test_index in cv.split(X, y):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

        # Fit model
        model.fit(X_train_cv, y_train_cv)

        # Binarize label (OvR)
        y_test_bin = label_binarize(y_test_cv, classes=np.unique(y))
        
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_cv)
        else:
            y_score = model.decision_function(X_test_cv)

        # Compute macro-average ROC
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr,
             label=f"{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (Multiclass OvR)")
plt.legend(loc="lower right")
plt.show()

```

    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.001040 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1281
    [LightGBM] [Info] Number of data points in the train set: 3539, number of used features: 104
    [LightGBM] [Info] Start training from score -1.136331
    [LightGBM] [Info] Start training from score -1.717974
    [LightGBM] [Info] Start training from score -0.693995
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000934 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1281
    [LightGBM] [Info] Number of data points in the train set: 3539, number of used features: 105
    [LightGBM] [Info] Start training from score -1.135451
    [LightGBM] [Info] Start training from score -1.717974
    [LightGBM] [Info] Start training from score -0.694561
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000952 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1272
    [LightGBM] [Info] Number of data points in the train set: 3539, number of used features: 104
    [LightGBM] [Info] Start training from score -1.135451
    [LightGBM] [Info] Start training from score -1.717974
    [LightGBM] [Info] Start training from score -0.694561
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000286 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 1276
    [LightGBM] [Info] Number of data points in the train set: 3539, number of used features: 103
    [LightGBM] [Info] Start training from score -1.135451
    [LightGBM] [Info] Start training from score -1.717974
    [LightGBM] [Info] Start training from score -0.694561
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] Found whitespace in feature_names, replace with underlines
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.000926 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1277
    [LightGBM] [Info] Number of data points in the train set: 3540, number of used features: 102
    [LightGBM] [Info] Start training from score -1.135734
    [LightGBM] [Info] Start training from score -1.716683
    [LightGBM] [Info] Start training from score -0.694844
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    


    
![png](output_78_1.png)
    



```python
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

# Inisialisasi list untuk confusion matrix dan dictionary untuk metrik
all_cms = []
tuned_metrics = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

# Evaluasi semua model yang sudah di-tune
for name, model in tuned_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    all_cms.append(cm)

    tuned_metrics["Model"].append(name)
    tuned_metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
    tuned_metrics["Precision"].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    tuned_metrics["Recall"].append(recall_score(y_test, y_pred, average='weighted'))
    tuned_metrics["F1 Score"].append(f1_score(y_test, y_pred, average='weighted'))

# Buat dataframe dari tuned_metrics
df_tuned_metrics = pd.DataFrame(tuned_metrics)

# Lanjut ke visualisasi
df_metrics = df_tuned_metrics.set_index("Model")

# Plot bar chart
ax = df_metrics.plot(kind='bar', figsize=(14, 6), colormap='viridis')
ax.set_ylabel('Score')
ax.set_title('Perbandingan Accuracy, Precision, Recall, dan F1 Score (Tuned Models)')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
```


    
![png](output_79_0.png)
    



```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Gunakan tuned_models, bukan models
n_models = len(tuned_models)
cols = 3
rows = int(np.ceil(n_models / cols))
vmax = max(cm.max() for cm in all_cms)

sns.set_style("white")
fig, axes = plt.subplots(rows, cols,
                         figsize=(cols * 3.5, rows * 3.5),
                         sharex=True, sharey=True)
axes = axes.flatten()

for idx, ((name, model), cm) in enumerate(zip(tuned_models.items(), all_cms)):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cbar=False, vmin=0, vmax=vmax, square=True,
                linewidths=0.5, linecolor='gray', cmap="viridis")
    ax.set_title(name, fontsize=10)

    row = idx // cols
    col = idx % cols

    ax.set_xlabel("Predicted" if row == rows - 1 else "", fontsize=8)
    ax.set_ylabel("True" if col == 0 else "", fontsize=8)

# Hapus axes kosong jika ada
for j in range(n_models, len(axes)):
    fig.delaxes(axes[j])

# Tambahkan colorbar global
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
norm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=vmax))
norm.set_array([])
fig.colorbar(norm, cax=cbar_ax, label="Count")

plt.suptitle("Confusion Matrices for Tuned Models", fontsize=14, y=1.02)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()

```

    C:\Users\Mumtaz\AppData\Local\Temp\ipykernel_29252\161404136.py:41: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
      plt.tight_layout(rect=[0, 0, 0.9, 1])
    


    
![png](output_80_1.png)
    

