import pandas as pd
import numpy as np

file_name = 'DATA/forensics/Forensics/eye+skin-color/eye.csv'
class_name = 'EYE-1a6'

df = pd.read_csv(file_name, delimiter=',', header=0, index_col=0)
df.columns = df.columns.str.strip()
df[class_name] = df[class_name].str.strip()
print(df)

y = df[class_name]
x = df.drop([class_name],axis=1)

oh = pd.DataFrame(index=x.index)
print(y)
print(x)
print(oh)

print(np.unique(df[df.columns].values))

for col in x.columns:
    if col[-2:] == '.1':
        c = col[:-2]
    else:
        c = col
    oh[c+'_A'] = np.zeros(len(x[col]))
    oh[c+'_C'] = np.zeros(len(x[col]))
    oh[c+'_G'] = np.zeros(len(x[col]))
    oh[c+'_T'] = np.zeros(len(x[col]))
    #oh[c+'_D'] = np.zeros(len(x[col]))

for col in x.columns:
    if col[-2:] == '.1':
        c = col[:-2]
    else:
        c = col
    x[col] = x[col].str.strip()
    oh[c+'_A'] = np.where(x[col]=='A', oh[c+'_A']+0.5, oh[c+'_A']+0)
    oh[c+'_C'] = np.where(x[col]=='C', oh[c+'_C']+0.5, oh[c+'_C']+0)
    oh[c+'_G'] = np.where(x[col]=='G', oh[c+'_G']+0.5, oh[c+'_G']+0)
    oh[c+'_T'] = np.where(x[col]=='T', oh[c+'_T']+0.5, oh[c+'_T']+0)
    #oh[c+'_D'] = np.where(x[col]=='D', oh[c+'_D']+0.5, oh[c+'_D']+0)

oh[class_name] = y
#oh = oh[oh[class_name] != 'NC']
'''oh[class_name] = oh[class_name].replace('1WHITE', 'Light')
oh[class_name] = oh[class_name].replace('2PALE', 'Light')
oh[class_name] = oh[class_name].replace('3BEIGE', 'Intermediate')
oh[class_name] = oh[class_name].replace('4LIG-BRW', 'Intermediate')
oh[class_name] = oh[class_name].replace('5MED-BRW', 'Dark')
oh[class_name] = oh[class_name].replace('6DRK-BRW', 'Dark')'''
#oh[class_name] = oh[class_name].replace('1AZ', '1Blue')
#oh[class_name] = oh[class_name].replace('2V', '2Intermediate')
#oh[class_name] = oh[class_name].replace('3M', '2Intermediate')
#oh[class_name] = oh[class_name].replace('4CC', '2Intermediate')
#oh[class_name] = oh[class_name].replace('5CE', '3DarkBrown')
#oh[class_name] = oh[class_name].replace('6PR', '3DarkBrown')
#print(oh)

oh.to_csv(file_name.replace('.csv', 'oh.csv'))