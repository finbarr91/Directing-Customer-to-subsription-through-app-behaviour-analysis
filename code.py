# Directing-Customer-to-subsription-through-app-behaviour-analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from dateutil import parser
dataset = pd.read_csv('appdata10.csv')
print('Head of the dataset:\n',dataset.head())
print('Tail of the dataset:\n', dataset.tail())
print('Description of the dataset:\n', dataset.describe())
print('Info of the dataset:\n', dataset.info())

# DATA CLEANING
dataset['hour']= dataset.hour.str.slice(1,3).astype('int') # the hour column is in a string format which looks like a date, so it is important it is converted to an int format for a better operation
print(dataset['hour'])

# PLOTTING
dataset2 = dataset.copy().drop(columns= ['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])
# creating a temporal dataset which is a copy of the original and dropping the list as shown
print(dataset2.head())

# HISTOGRAMS
plt.suptitle('Histogram of Numerical Columns', fontsize = 10)
for i in range(1, dataset2.shape[1]+1):
    plt.subplot(3,3,i)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i-1])
    vals = np.size(dataset2.iloc[:, i-1].unique())

    plt.hist(dataset2.iloc[:, i -1], bins=vals, color='#3F5D7D')
    
plt.tight_layout()
plt.show()

# CORRELATION VARIABLE
dataset2.corrwith(dataset['enrolled']).plot.bar(figsize = (20,10),
                                               title = 'Correlation Variable',
                                               fontsize = 15, rot =45,
                                                grid= True)
plt.tight_layout()
plt.show()

# CORRELATION MATRIX :
# This shows how each individual feature relates to another(linearly, inversely etc)
# The essence of checking for the correlation matrix is that an assumption in ML which is every features must be independent of each other
# so as to give a good result of the model. If we have dependent features, then the model will produce an accurate result

sns.set(style = 'white', font_scale =2)

# Compute the correlation matrix
corr = dataset2.corr()

#Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f,ax = plt.subplots(figsize = (18,15))
f.suptitle('Correlation Matrix', fontsize =10)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220,10, as_cmap = True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask = mask, cmap=cmap, vmax = .3, center =0, square = True, linewidths=.5, cbar_kws={'shrink': .5})
plt.tight_layout()
plt.show()

# FEATURE ENGINEERING
print(dataset.dtypes)
dataset['first_open']=[parser.parse(row_data) for row_data in dataset['first_open']] # This will convert every row to a date-time object
print("\n dataset['first_open']\n",dataset['first_open'])
dataset['enrolled_date'] = [parser.parse(row_data) if isinstance(row_data,str) else row_data for row_data in dataset['enrolled_date']]
print("\ndataset['enrolled_date']\n",dataset['enrolled_date'])

dataset['difference'] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[h]') # to get the duration taken for the constumer to subscribe and also convert to hours
plt.hist(dataset['difference'].dropna(), color = '#3F5D7D')
plt.title('Distribution of Time-Since-Enrolled')
plt.show()

dataset['difference'] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[h]') # to get the duration taken for the constumer to subscribe and also convert to hours
plt.hist(dataset['difference'].dropna(), color = '#3F5D7D', range = [0,100])
plt.title('Distribution of Time-Since-Enrolled')
plt.show()

dataset.loc[dataset.difference >48, 'enrolled'] =0
dataset = dataset.drop(columns=['difference', 'enrolled_date', 'first_open'])

# FORMATTING THE SCREEN_LIST FIELD
top_screens = pd.read_csv('top_screens.csv').top_screens.values
print("\n Top screens:\n", top_screens)
dataset['screen_list'] = dataset.screen_list.astype(str) + ','
for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset['screen_list'] = dataset.screen_list.str.replace(sc+ ",", "")

dataset["Other"] = dataset.screen_list.str.count(",")
dataset = dataset.drop(columns = ['screen_list'])
print('\nDataset:\n', dataset.columns)
print('\n Dataset Head:\n', dataset.head())


# Funnels
savings_screens = ['Saving1',
                   'Saving2',
                   'Saving2Amount',
                   'Saving4',
                   'Saving5',
                   'Saving6',
                   'Saving7',
                   'Saving8',
                   'Saving9',
                   'Saving10']

dataset['SavingsCount'] = dataset[savings_screens].sum(axis=1)
dataset = dataset.drop(columns= savings_screens)

cm_screens = ['Credit1',
              'Credit2',
              'Credit3',
              'Credit3Container',
              'Credit3Dashboard']

dataset['CMCount'] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)

cc_screens = ['CC1',
              'CC1Category',
              'CC3']

dataset['CCCount'] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)

loan_screens = ['Loan',
                'Loan2',
                'Loan3',
                'Loan4']

dataset['LoansCount'] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)

print(dataset.head())
print(dataset.describe())
print(dataset.info())

dataset.to_csv('new_appdata10.csv', index = False)















