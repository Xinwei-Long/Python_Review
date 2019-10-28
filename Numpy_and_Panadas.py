##fixed type arrays
import array
L=list(range(10))
A=array.array("i",L)#integer array


##high dimensional arrays
import numpy as np
np.array([range(i,i+3) for i in [2,4,6]]) #inner list serves as row

##build arrays from scratch
np.ones((3,5),dtype=int)#can always sepcify type

##key categories of arrays: attributes, indexing, slicing, reshaping, joining and splitting
#attributes
np.random.seed(0)
x3=np.random.randint(10,size=(3,4,5))
print(x3.ndim)
print(x3.shape)
print(x3.size)

#indexing
x3[0,0,0]=3.14#caveat: results be truncated since original array is fixed to int

#slicing
#x[start:stop:step]
x=np.arange(10)
x=x[5::-2]# reversed every other from index 5

x2=np.random.randint(10,size=(3,4))
x2[:,0]#first column

x2_copy=x2[:2,:2].copy()#copy array

#reshaping
x=np.array([1,2,3])
x.reshape((1,3))
x[np.newaxis,:]#same as above

x.reshape((3,1))#same as above
x[:,np.newaxis]

#concatenation
x=np.array([1,2,3])
y=np.array([4,5,6])
np.concatenate([x,y])

np.vstack([x,y])#vertically stack
np.hstack([x,y])#horizontally stack
#nstack: stack along 3rd axis

#splitting
x=[1,2,3,4,5,6,7,8]
x1,x2,x3=np.split(x,[3,5])

grid=np.arange(16).reshape((4,4))
upper,lower=np.vsplit(grid,[2])#vertically split
left,right=np.hsplit(grid,[2])#horizontally split

###vertorized calculation through Ufunc
#specify output
x=np.arange(5)
y=np.zeros(10)
np.power(2,x,out=y[::2])

##aggregate
#reduce
x=np.arange(1,6)
np.add.reduce(x)

#accumulate--store intermediates
np.add.accumulate(x)

###aggregations
#use np.sum np.min quicker than direct sum, min, max

#multidimentional aggregates
m=np.random.random((3,4))
m.sum(axis=0)#sum of each col
m.sum(axis=1)#sum of each row

#broadcasting vectorized
#eg. plot 2 dim function
x=np.linspace(0,5,50)#50 steps from 0 to 5
y=np.linspace(0,5,50)[:,np.newaxis]

z=np.sin(x)**10+np.cos(10+y*x)*np.cos(x)

import matplotlib.pyplot as plt
plt.imshow(z,origin='lower',extent=[0,5,0,5],cmap='viridis')
plt.colorbar()

##boolean arrays
x=np.random.randint(10,size=(3,4))
np.sum(x<6)#count entries less than 0
np.sum(x<6,axis=1)#in each row
np.any(x<6)
np.all(x<6)

##bool operators

##fancy indexing
x=np.arange(12).reshape((3,4))
row=np.array([0,1,2])
mask=np.array([1,0,1,0],dtype=bool)
x[row[:,np.newaxis],mask]

##modify values with fancy indexing
x=np.arange(10)
i=np.array([2,1,8,4])
x[i]+=1#unexpected results
np.add.at(x,i,1)

###Sort arrays
#simply sort:np.sort and np.argsort
x=np.array([2,1,4,3,5])
np.sort(x)
x.sort()#sort in-place
i=np.argsort(x)#return index
x[i]

#sorting along rows or columns
rand=np.random.RandomState(42)
X=np.random.randint(0,10,(4,6))
np.sort(X,axis=0)#along col
np.sort(X,axis=1)#along row

#partial sort: partition
x=np.array([7,2,3,1,6,5,4])
np.partition(x,3)#smallest K to the left
np.partition(X,3,axis=1)

#sorting example: K-Nearest Neighbors
K=rand.rand(10,2)
difference=K[:,np.newaxis,:]-K[np.newaxis,:,:]#two arrays differ in the 2-nd dimension, where broadcasting occurs
sq_difference=difference**2
dist_sq=sq_difference.sum(axis=-1) #in here, axis=0 means along list, axis=1 along col, axis=0 along row

dist_sq.diagonal()
nearest=np.argsort(dist_sq,axis=1)#index of K-nearest distance
k=2
nearest_partition=np.argpartition(dist_sq,k+1,axis=1)

##structured arrays-created by dictionary method
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
#conpound type specification to specify the empty array
data=np.zeros(4,dtype={'names':('name','age','weight'),'formats':('U10','i4','f8')})
data['name']=name
data['age']=age
data['weight']=weight
#bool search:names of those whose ages<30
data[data['age']<30]['name']

#structure arrays--created by turple
dtype=np.dtype([('name','S10'),('age','i4'),('weight','f8')])
np.dtype('S10,i4,f8')
data=np.array([('A','22','22')],dtype=dtype)

#advance compound
tp=np.dtype([('id','i8'),('mat','f8',(3,3))])
x=np.zeros(1,dtype=tp)

#recarray
data_rec=data.view(np.recarray)
data_rec.age

###################Pandas
#use dictionary to construct series
import pandas as pd
population_dict = {'California': 38332521,
'Texas': 26448193,
'New York': 19651127,
'Florida': 19552860,
'Illinois': 12882135}
population=pd.Series(population_dict)

#use dictionary to construct dataframe
area_dic={'California': 423967, 'Texas': 695662, 'New York': 141297,
'Florida': 170312, 'Illinois': 149995}
area=pd.Series(area_dic)
states=pd.DataFrame({'population':population,'area':area})
states.index

##construct dataframe
#from series
pd.DataFrame(population,columns=['population'])
#from list of dic-key as column=[],values as values
data=[{'a':i,'b':i*2} for i in range(3)]
pd.DataFrame(data)
#missing value marked as NaN
pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
#from dis of series
pd.DataFrame({'population':population,'area':area})
#from 2-dim np arrays
pd.DataFrame(np.random.rand(3,2),columns=['foo','bar'],index=['a','b','c'])
#from structured array
pd.DataFrame(np.zeros(3,dtype=[('A','i8'),('B','f8')]))

#Pandas index object
indA = pd.Index([1, 3, 5, 7, 9])
indB=pd.Index([2,3,5,7,11])
indA & indB #intersection
indA | indB #union
indA ^ indB #symmetric diff

###data selection on Series
data = pd.Series([0.25, 0.5, 0.75, 1.0],
index=['a', 'b', 'c', 'd'])
#Series as dic
data['b']
data.keys()
list(data.items())
data['e']=1.25#extend the Series like dic
#Series as 1-dim array
data['a':'c']#final index is included
data[0:2]#final index is excluded
data[(data>0.3)&(data<0.8)]#masking

#indexers: loc,iloc, and ix
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
#loc:always refer to explicit index
data.loc[1]
data.loc[1:3]

#iloc:always refer to implicit
data.iloc[1]
data.iloc[1:3]

##Data selection on DF
#DF as dic of Series objects
area = pd.Series({'California': 423967, 'Texas': 695662,
'New York': 141297, 'Florida': 170312,
'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
'New York': 19651127, 'Florida': 19552860,
'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})

data['area']#dic-styled
data.area#attribute-styled (caveat: if the column names conflict
# with methods of the DataFrame, this attribute-style access is not possible
data['density']=data['pop']/data['area']

#DF as 2-dim array
data.values[0]#first row of array
data.T#transpose

data.iloc[:3, :2] #implicit indexing
data.loc[:'Illinois',:'pop'] #explicit indexing
data.ix[:3,:'pop'] #hybrid, caveat: if the column names conflict
# with methods of the DataFrame, this attribute-style access is not possible

data.loc[data.density>100,['pop','density']]#mask and fancy index

###Operation on data using Pandas
##index preservation
import numpy as np
import pandas as pd
rng=np.random.RandomState(42)
ser=pd.Series(rng.randint(0,10,4))
df = pd.DataFrame(rng.randint(0, 10, (3, 4)),columns=['A', 'B', 'C', 'D'])
np.exp(ser)

##index assignment
#in series
area=pd.Series({'Alaska': 1723337, 'Texas': 695662,'California': 423967}, name='area')
population=pd.Series({'California': 38332521, 'Texas': 26448193,'New York': 19651127}, name='population')
population/area #NaN for missing match of index

A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A+B #add by matched index
A.add(B,fill_value=0) #explicit specification of the fill value for any elements in A or B that might be missing

#in DF
#indices are aligned correctly irrespective of their order in the two objects
A=pd.DataFrame(rng.randint(0,10,(2,2)),columns=list('AB'))
B=pd.DataFrame(rng.randint(0,20,(3,3)),columns=list('BCA'))
fill=A.stack().mean() #mean of all values in A
A.add(B,fill_value=fill) #replace all missing values with fill

##operations between DF and Series
A=rng.randint(10,size=(3,4))
df=pd.DataFrame(A,columns=list("QRST"))
df-df.iloc[0] #substract row-wise
df.subtract(df['R'],axis=0) #subtract col-wise

halfrow=df.iloc[0,::2] #will assign index and col names

##Dealing missing values
#in Pandas, NaN and None are interchangable
#detecting null values
data=pd.Series([1,np.nan,'hello',None])
data.isnull() #boolean masks, which can be used as Series and DF index
data[data.notnull()]

#dropping null values
data.dropna() #drop all rows with null (NaN, None)

df=pd.DataFrame([[1, np.nan, 2],
[2, 3, 5],
[np.nan, 4, 6]])

df.dropna(axis='columns') #drop all cols with null (NaN, None)

df[3]=np.nan #add one row of nan
df.dropna(axis='columns',how='all') #only drop rows/columns that are all null values
#how='any'--any row or column (depending on the axis keyword)containing a null value will be dropped

df.dropna(axis='rows',thresh=3) #at least 3 non-null values should be kept

#filling null values
df.fillna(method='ffill',axis='rows') #ffill:fill the NaN with forward value; bfill: fill the NaN with backward value
#if a previous value is not available during a forward fill, the NA value remains.

###Hierarchical indexing (high-dimension indexing)
index = [('California', 2000), ('California', 2010),
('New York', 2000), ('New York', 2010),
('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
18976457, 19378102,
20851820, 25145561]
index=pd.MultiIndex.from_tuples(index)
pop = pd.Series(populations, index=index)
pop=pop.reindex(index)
pop[:,2010]

#multiidenx as extra dimentsion
pop_df=pop.unstack()

##create multiIndex
#implicit
df = pd.DataFrame(np.random.rand(4, 2),
index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
columns=['data1', 'data2'])

data = {('California', 2000): 33871648,
('California', 2010): 37253956,
('Texas', 2000): 20851820,
('Texas', 2010): 25145561,
('New York', 2000): 18976457,
('New York', 2010): 19378102}
pd.Series(data)

#explicit
#from arrays
pd.MultiIndex.from_arrays([['a','a','b','b'],[1,2,1,2]])
#from turples
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
#from prodcut
pd.MultiIndex.from_product([['a','b'],[1,2]])
#directly
pd.MultiIndex(levels=[['a','b'],[1,2]], labels=[[0,0,1,1],[0,1,0,1]])

##name of index
pop.index.names=['state','year']

##multiindex for cols
index=pd.MultiIndex.from_product([[2013,2014],[1,2]],names=['year','visit'])
columns=pd.MultiIndex.from_product([['Bob','Guido','Sue'],['HR','Temp']],names=['subject','type'])
data=np.round(np.random.randn(4,6),1)
data[:,::2] *=10
data +=37
health_data=pd.DataFrame(data,index=index,columns=columns)
health_data['HR']

#slicing in Series--take lower level index as added dimention
#note rows are primary in Series slicing (when not using ix, iloc, loc)
pop.loc[:,2000]
pop[['California', 'Texas']] #same level, use list
pop['California', 2000]#different levels

#slicing in Series--take lower level index as added dimention
#note cols are primary in Series slicing (when not using ix, iloc, loc)
health_data['Guido', 'HR'] #different levels
health_data.loc[:, ('Bob', 'HR')]
#note: indexing within turples ('Bob', 'HR') may lead to syntax error
#if want to slice health_data.loc[(:, 1), (:, 'HR')]
idx=pd.IndexSlice
health_data.loc[idx[:,1],idx[:,'HR']]

###Rearranging multi-index
##sorted and unsorted indices--some slicing require levels to be sorted
index=pd.MultiIndex.from_product([['a','c','b'],[1,2]])
data=pd.Series(np.random.rand(6),index=index)
data.index.names=['chr','int']
data=data.sort_index()
data['a':'b']


##rearranging hierarchical
#stacking and unstacking
pop.unstack(level=0)#make index level 1 to col
pop.unstack(level=1)#make index level 2 to col

#index setting and resetting-turn index into columns
pop.index.names=['State','Year']
pop_flat=pop.reset_index(name='population')#Series become DF
pop_flat.set_index(['State','Year'])#Become Series

##Data Aggregations on multi-index
data_mean=health_data.mean(level='year') #level on row
data_mean=health_data.mean(axis=1,level='type') #level on col

###combing datasets_concat and append
def make_df(cols, ind):
#"""Quickly make a DataFrame"""
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)

df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
df3 = make_df('AB', [0, 1])
df4 = make_df('AB', [0, 1])
print(pd.concat([df1,df2])) #by default along the row
print(pd.concat([df3, df4], axis=1)) #must have the same row index

#pd.concat remain duplicated indices
y=make_df('AB',[0,1])
x=make_df('AB',[0,1])
print(pd.concat([x,y]))

#catching repeated index as error
try:
    pd.concat([x,y],verify_integrity=True)
except ValueError as e:
    print('ValueError:', e)

#ingnore index
pd.concat([x,y],ignore_index=True)
#add multi index keys
print(pd.concat([x,y],keys=['x','y']))

#Concatenation with joins
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
print(pd.concat([df5, df6])) #full join, by defualt join='outer', might be NaN in joining
print(pd.concat([df5, df6],join='inner')) #intersection of columns
print(pd.concat([df5, df6],join_axes=[df5.columns])) #directly specify columns

#concatenatioin with append
print(df1.append(df2)) #ignore original index by default, create new object
#advantage of append: create data buffer, but plan to do multiple
#append operations, it is generally better to build a list of DataFrames and pass them all
#at once to the concat() function.

###combing datasets: merge and join
##Category of joins--one-to-one,many-to-one,many-to-many
#on-to-one
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
'hire_date': [2004, 2008, 2012, 2014]})
df3=pd.merge(df1,df2) #dicard index, use column 1 in 2 datasets as join_axis

#many-to-one-one of the two key columns contains duplicate
#entries. For the many-to-one case, the resulting DataFrame will preserve those duplicate
#entries as appropriate.
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
'supervisor': ['Carly', 'Guido', 'Steve']})
print(pd.merge(df3,df4))

#many-to-many
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
'Engineering', 'Engineering', 'HR', 'HR'],'skills': ['math', 'spreadsheets', 'coding', 'linux',
'spreadsheets', 'organization']})
print(pd.merge(df1,df5))

#specification of merge key
print(pd.merge(df1,df2,on='employee')) #speicify key col or list of key cols

df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
'salary': [70000, 80000, 120000, 90000]})
print(pd.merge(df1, df3, left_on="employee", right_on="name")) #if cols contain same type of information have different names
pd.merge(df1, df3, left_on="employee", right_on="name").drop('name',axis=1)#drop duplicated col

#merge by index
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print(pd.merge(df1a, df2a, left_index=True, right_index=True))
print(df1a.join(df2a)) #merge by index

#mixed merge-by col and index
print(pd.merge(df1a,df3,left_index=True,right_on='name'))

##Specifying Set Arithmetic for Joins
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
'food': ['fish', 'beans', 'bread']},
columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
'drink': ['wine', 'beer']},columns=['name', 'drink'])
#how: outer:full join with missing values as NaN
#inner:appear in keys in both cols
#left: appear in left col
print(pd.merge(df6,df7,how='left'))

##Overlapping col values:suffixes keyword
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
'rank': [3, 1, 4, 2]})

print(pd.merge(df8, df9, on="name",suffixes=['_L','_R']))

###Aggregation and Grouping
import seaborn as sns
rng=np.random.RandomState(42)
planets=sns.load_dataset('planets')

df = pd.DataFrame({'A': rng.rand(5),'B': rng.rand(5)})
df.mean(axis='columns') #operation within each row
#statistics for each col
planets.dropna().describe()

##Groupby: split, apply, combine
#split:breaking up and grouping a DataFrame depending on the value of the specified key.
#apply:computing some function, usually an aggregate, transformation, or filtering, within the individual groups
#combine: merges the results of these operations into an output array
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
'data': range(6)}, columns=['key', 'data'])
df.groupby('key') #not a DF object, but DataFrameGroupBy object, and no calculation is done
df.groupby('key').sum() #split-apply-combine on Series data

planets.groupby('method')['orbital_period'].median()

#iteration over groups
for (method,group) in planets.groupby('method'):
    print('{0:30s} shape={1}'.format(method,group.shape))

#dispatch: operations on Series & DF can be applied on groupby objects
planets.groupby('method')['year'].describe()

##aggregate, filter, transform. apply
rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
'data1': range(6),
'data2': rng.randint(0, 10, 6)},
columns = ['key', 'data1', 'data2'])

#aggregate: take list or dictionary as inputs
df.groupby('key').aggregate(['min',np.median,max]) #don't have to specify input in the function
df.groupby('key').aggregate({'data1':'min','data2':'max'})

#filter
def filter_func(x):
    return x['data2'].std() > 0.5
print(df.groupby('key').filter(filter_func)) #don't have to specify input in the function

#transformation-output is the same shape as input
df.groupby('key').transform(lambda x: x-x.mean())

#apply-apply an arbitrary function to the group results
#function should take a DataFrame, and return either a Pandas object (e.g., DataFrame, Series) or a scalar
def norm_by_data2(x):
    #x is a DF of grouped values
    x['data1']/x['data2'].sum()
    return x

##different ways of specifying split key
#from array, list or index
L=[0,1,0,1,2,0] #provide level of index as split key, same length as DF
print(df.groupby(L).sum())

#from dictionary, dic maps index to group keys
mapping={'A':'vowel','B':'consonant','C':'consonant'}
df2=df.set_index('key')
print(df2.groupby(mapping).sum())

#from any function
print(df2.groupby(str.lower).mean())

#list of valid keys (list, array,dic, function)-multi index
print(df2.groupby([str.lower, mapping]).mean())

#eg.find the number of planets found by each method across decades
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack(level=1).fillna(0)

####Pivot tables
import pandas as pd
import numpy as np
import seaborn as sns
titanic=sns.load_dataset('titanic')

##pivot by hand
titanic.groupby('sex')[['survived']].mean()
titanic.groupby('sex')['survived'].mean() #same result, but no col name
titanic.groupby(['sex','class'])['survived'].aggregate('mean').unstack()

##pivot table syntax
titanic.pivot_table('survived',index='sex',columns='class')

#multi level pivot tables
#use pd.cut to bin (range) the variable
#multi level at row
age=pd.cut(titanic['age'],[0,18,80]) #dtype=category
titanic.pivot_table('survived',['sex',age],'class')

#mutlti level at col
fare=pd.qcut(titanic['fare'],2) #bin by quantile
titanic.pivot_table('survived',['sex',age],[fare,'class'])

##additional options
#aggfunc: can be specified as a dictionary mapping a column to any of the above desired options, by default 'mean'
titanic.pivot_table(index='sex',columns='class',aggfunc={'survived':sum,'fare':'mean'})#not specify 'value' since specified in aggfunc
#margin: margins: total along each group
titanic.pivot_table(values='survived',index='sex',columns='class',margins=True,margins_name='Ave_All')


####Vectorized String operations
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
'Eric Idle', 'Terry Jones', 'Michael Palin'])
#pandas string methods-str.xxx
#using regular expressions
#Miscellaneous methods
#vectorized item access and slicing
monte.str[0:3]
monte.str.split().str.get(-1) #get the last name splitted by " "

#indicator variables-get_dummies, when your data has a column containing some sort of coded indicator
full_monte = pd.DataFrame({'name': monte,
'info': ['B|C|D', 'B|D', 'A|C', 'B|D', 'B|C',
'B|C|D']})
full_monte['info'].str.get_dummies('|')

#########Time Series
###Dates and time
from datetime import datetime
datetime(year=2015,month=7,day=4)
from dateutil import parser
date= parser.parse('4th of July,2015')

#Typed arrays of times:Numpy's datetime64
date=np.array('2015-07-04',dtype=np.datetime64)
date+np.arange(12) #vectorized
date=pd.to_datetime('4th of July, 2015')
date.strftime('%A') #return the week

##Pandas Time Series: Indexing by time
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
'2015-07-04', '2015-08-04'])
data=pd.Series([0,1,2,3],index=index)
#slicing
data['2015']

##Panadas Time Series data structure
#to_datetime: Passing a single date to pd.to_datetime() yields a Timestamp; passing a series of dates by
#default yields a DatetimeIndex
dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
'2015-Jul-6', '07-07-2015', '20150708'])
dates
#to_period: Any DatetimeIndex can be converted to a PeriodIndex with the to_period() function
#with the addition of a frequency code
dates.to_period('M')
dates-dates[0] #create index by delta

#regular sequences: pd.date_range(),pd.period_range(),pd.timedelta_range()
pd.date_range('2015-07-03', '2015-07-10')
pd.date_range('2015-07-03', periods=4, freq='H') #specifiy starting point
pd.date_range('2015-07-03', periods=8, freq='QS') #quarter start

pd.period_range('2015-07', periods=8, freq='M')
pd.timedelta_range(0, periods=10, freq='H')
pd.timedelta_range(0,periods=9,freq='2H30T')

##Resampling, Shifting, and Windowing
#resample(),asfreq(), fill NaN by 0 by default
time_series=np.random.randint(0,10,(10000,2))
data=pd.DataFrame(time_series,columns=['A','B'],index=pd.date_range('2015-07-03', periods=10000, freq='D'))
dataA=data['A']
#resample reports the average of the previous year, while asfreq reports the value at the end of the year.
dataA.asfreq('BA').plot(style='--')
dataA.resample('BA').mean().plot(style=':')
#interpolation by asfreq
import  matplotlib.pyplot as plt
fig, ax = plt.subplots(2, sharex=True)
dataA=dataA.iloc[:10]
dataA.asfreq('BA', method='bfill').plot(ax=ax[1], style='-o')#backward
dataA.asfreq('BA', method='ffill').plot(ax=ax[1], style='--o')#forward

#time shifts:
#shift():shift data; tshift():shift index
fig, ax = plt.subplots(3, sharey=True)
dataA = dataA.asfreq('D', method='pad')

dataA.plot(ax=ax[0])
dataA.shift(10).plot(ax=ax[1])
dataA.tshift(10).plot(ax=ax[2])

#rolling windows
#eg. one year rolling statistics
rolling=dataA.rolling(365,center=True)
data=pd.DataFrame({'input':dataA,'one-year rolling mean':rolling.mean(),'one-year rolling std':rolling.std()})
ax=data.plot(style=['-','--',':'])
ax.lines[0].set_alpha(0.3)

#get the formation of date index
dataA.index.time
dataA.indx.dayofweek
dataA.index.weekday

###High-performance Pandas: eval(), query()
#eval():string expressions to efficiently compute operations using DataFrames
rng = np.random.RandomState(42)
df1,df2,df3,df4,df5=(pd.DataFrame(rng.randint(0,1000,(100,3))) for i in range(5))
#arithmetic operators
result1 = -df1 * df2 / (df3 + df4) - df5
result2 = pd.eval('-df1 * df2 / (df3 + df4) - df5')
np.allclose(result1, result2) #same result
#comparison: include chained comparision
result1 = (df1 < df2) & (df2 <= df3) & (df3 != df4)
result2 = pd.eval('df1 < df2 <= df3 != df4')
#Bitwise:& and |
result2=pd.eval('(df1<0.5) & (df2<0.5) | (df3<df4)')
#use of the literal and and or in Boolean expressions:
result2=pd.eval('(df1<0.5) and (df2<0.5) or (df3<df4)')
#Object attributes and indices.
result2 = pd.eval('df2.T[0] + df3.iloc[1]')

#DF.eval() for columns-wise operation
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
result2=df.eval('(A+B)/(C-1)')

#assignment of columns
df.eval('D=(A+B)/C',inplace=True) #create new column
df.eval('D=(A+B)/(C-1)',inplace=True) #modify column

##local variable assignment
column_mean=df.mean(1)
result2=df.eval('A+@column_mean') #marks a variable name rather than a column name

##DF.query()
result2=df.query('A<0.5 & B<0.5')
#local varaible
Cmean = df['C'].mean()
result1 = df[(df.A < Cmean) & (df.B < Cmean)]
result2 = df.query('A < @Cmean and B < @Cmean')

########################################## Visualization with Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
x = np.linspace(0, 10, 100)
fig = plt.figure() #set one empty plot (plot figure)
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')

##Saving figures to File
fig.savefig('my_picture.png')
fig.canvas.get_supported_filetypes() #check the supported type
#display what is saved
from IPython.display import Image
Image('my_picture.png')

##matlab-style interface
plt.figure() # create a plot figure
# create the first of two panels and set current axis
plt.subplot(2, 1, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))
# create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x));
#problem:static, cannot get back to the first subplot easily

##Object-oriented interface
# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)
# Call plot() method on the appropriate object
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))

##simple line plots
plt.style.use('seaborn-whitegrid')
fig=plt.figure()
ax=plt.axes()

x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x))
