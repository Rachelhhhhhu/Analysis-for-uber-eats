import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('4.20.csv')

# 所有的州的avg
avg_by_state = round(df.groupby('R_STATE')['Restaurant_Rating'].mean(),1)
sorted_df = avg_by_state.sort_values(ascending=False)
print(sorted_df)

#top 5 ：MN, NY, CA, NC, HV

# # ## 证明大于95%的概率可以说明 高评分与地区有关系 证明到底评分的分类正确
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

grouped = df.groupby('R_STATE')
means = grouped['boo'].mean()

# ANOVA检验
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
formula = 'boo ~ C(R_STATE)'
model = ols(formula, data=df).fit()
aov_table = anova_lm(model, typ=2)

print('Means:')
print(means)
print('')
print('ANOVA Table:')
print(aov_table)

#相关系数corr
import seaborn as sns
import matplotlib.pyplot as plt

corr = df[['BUSINESS_HOUR', 'Restaurant_Rating', 'PRODUCT_PRICE']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


# 散点图
plt.scatter(df['NY_STATE'], df['Restaurant_Rating'])

plt.title('Restaurant_Rating vs. NY_STATE')
plt.xlabel('NY_STATE')
plt.ylabel('Restaurant_Rating')

plt.show()

import csv
import seaborn as sns
import matplotlib.pyplot as plt

rating1 = []
rating2 = []
rating3 = []
rating4 = []
rating5 = []
rating6 = []

with open('4.20.csv', 'r') as f:
    reader = csv.reader(f)
    for col in reader:
        if col[7] == 'CA':
            rating_str = col[2]
            if rating_str != '':
                rating1.append(float(rating_str))
        elif col[7] == 'IL':
            rating_str = col[2]
            if rating_str != '':
                rating2.append(float(rating_str))
        elif col[7] == 'NY':
            rating_str = col[2]
            if rating_str != '':
                rating3.append(float(rating_str))
        elif col[7] == 'NC':
            rating_str = col[2]
            if rating_str != '':
                rating4.append(float(rating_str))
        elif col[7] == 'NV':
            rating_str = col[2]
            if rating_str != '':
                rating5.append(float(rating_str))
grouped = df.groupby('R_STATE')['Restaurant_Rating'].mean()
# total 
fig, ax = plt.subplots()
sns.boxplot(data=[rating1, rating2, rating3, rating4, rating5, rating6], ax=ax, palette=['skyblue', 'pink', 'lightgreen', 'orange', 'c','r'])
ax.set(title='Multiple Boxplots', xlabel='CA, IL, NY, NC, NV',ylabel='Rating')
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('4.20.csv')

grouped = data.groupby('R_STATE')['Restaurant_Rating'].mean()

fig, ax = plt.subplots()
sns.boxplot(data=[data[data['R_STATE'] == 'CA']['Restaurant_Rating'].dropna(),
                  data[data['R_STATE'] == 'IL']['Restaurant_Rating'].dropna(),
                  data[data['R_STATE'] == 'NY']['Restaurant_Rating'].dropna(),
                  data[data['R_STATE'] == 'NC']['Restaurant_Rating'].dropna(),
                  data[data['R_STATE'] == 'NV']['Restaurant_Rating'].dropna(),
                  grouped.values],
            ax=ax,
            palette=['skyblue', 'pink', 'lightgreen', 'orange', 'c', 'r'])
ax.set(title='Multiple Boxplots', xlabel='CA, IL, NY, NC, NV, Total', ylabel='Rating')
plt.show()



# COUNT CATEGORY 
import numpy as np

chinese = np.nansum(np.nan_to_num(df['CATEGORY_TYPE'].str.contains('Chinese')))
japanese = np.nansum(np.nan_to_num(df['CATEGORY_TYPE'].str.contains('Japanese')))
mexican = np.nansum(np.nan_to_num(df['CATEGORY_TYPE'].str.contains('Mexican')))
health = np.nansum(np.nan_to_num(df['CATEGORY_TYPE'].str.contains('Health')))
seafood = np.nansum(np.nan_to_num(df['CATEGORY_TYPE'].str.contains('Seafood')))
korean = np.nansum(np.nan_to_num(df['CATEGORY_TYPE'].str.contains('Korean')))
brunch = np.nansum(np.nan_to_num(df['CATEGORY_TYPE'].str.contains('Brunch')))
italian = np.nansum(np.nan_to_num(df['CATEGORY_TYPE'].str.contains('Italian')))
american = np.nansum(np.nan_to_num(df['CATEGORY_TYPE'].str.contains('American')))
india = np.nansum(np.nan_to_num(df['CATEGORY_TYPE'].str.contains('India')))

print(f"chinese: {chinese}")
print(f"japanese: {japanese}")
print(f"mexican: {mexican}")
print(f"health: {health}")
print(f"seafood: {seafood}")
print(f"korean: {korean}")
print(f"brunch: {brunch}")
print(f"italian: {italian}")
print(f"american: {american}")
print(f"india: {india}")

import matplotlib.pyplot as plt

counts = [chinese, japanese, mexican, health, seafood, korean, brunch, italian, american, india]
categories = ['Chinese', 'Japanese', 'Mexican', 'Health', 'Seafood', 'Korean', 'Brunch', 'Italian', 'American', 'India']

fig, ax = plt.subplots(figsize=(10,5))
ax.barh(categories, counts)

ax.set_xlabel('Number of restaurants')
ax.set_ylabel('Category')
ax.set_title('Number of restaurants in each category')

plt.show()

# reorder info rate
import numpy as np
import matplotlib.pyplot as plt

reorder_info = df['REORDER_INFO_1'].to_numpy()

reorder_count = np.bincount(reorder_info)
reorder_percent = reorder_count / len(reorder_info)

labels = ['Null', 'Reorder']
colors = ['#ff9999', '#66b3ff']
fig1, ax1 = plt.subplots()
ax1.pie(reorder_percent, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
plt.title('Reorder Info')
plt.show()


#不同州business的直方图：
import matplotlib.pyplot as plt

state_duration = df.groupby('R_STATE')['BUSINESS_HOUR'].mean()
state_names = state_duration.index.tolist()
state_avg_duration = state_duration.tolist()

plt.barh(state_names, state_avg_duration, color='skyblue')
plt.xlabel('Average Business Duration')
plt.ylabel('R_STATE')
plt.title('Distribution of Average Business Duration by State')
plt.show()


不同州的avg price
import matplotlib.pyplot as plt

state_duration = df.groupby('R_STATE')['PRODUCT_PRICE'].mean()
state_names = state_duration.index.tolist()
state_avg_duration = state_duration.tolist()

plt.barh(state_names, state_avg_duration, color='skyblue')
plt.xlabel('Average Product Price')
plt.ylabel('R_STATE')
plt.title('Distribution of Average Product Price by State')
plt.show()

# COUNT STATE 
import matplotlib.pyplot as plt
state_count = df['R_STATE'].value_counts()

plt.barh(state_count.index, state_count.values)
plt.xlabel('R_STATE')
plt.ylabel('Number of Restaurants')
plt.title('Distribution of Restaurants by State')
plt.xticks(rotation=90)
plt.show()


