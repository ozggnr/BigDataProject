import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.style as style
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
os.makedirs('plots', exist_ok=True)
#Data preparation
data = pd.read_csv('books.csv', error_bad_lines=False)
data.rename({'  num_pages': 'page_number', 'language_code': 'language', 'text_reviews_count': 'text_count'}, axis=1, inplace=True)
df = data.drop(['isbn', 'isbn13'], axis=1)
df['place'] = df['authors'].str.find('/')
for i in df.index:
    if df['place'][i] == -1:
        continue
    else:
        df.loc[i, 'authors'] = df.loc[i, 'authors'][:df.loc[i, 'place']]
dd = df.drop('place', axis=1)
dd['date'] = dd['publication_date'].str.rfind('/')
for i in dd.index:
    if dd['date'][i] == -1:
        continue
    else:
        dd.loc[i, 'publication_date'] = dd.loc[i, 'publication_date'][dd.loc[i, 'date']+1:]
dd['publication_date'] = dd['publication_date'].astype(int)
dd = dd.drop(['date', 'publisher', 'language', 'title'], axis=1)

dd = dd[dd['ratings_count'] != 0]
dd = dd[dd['ratings_count'] < 900000]
dd = dd[dd['page_number'] < 2000]
dd['date'] = pd.cut(dd['publication_date'], bins=[1900, 2002, 2020], labels=['old', 'new'], right=True, include_lowest=True)
#Authors with average ratings count who published book after 2003
new_author = dd[['authors', 'date', 'ratings_count']][dd['date'] == 'new']
new_author_ratings = new_author[['authors', 'ratings_count']].groupby('authors').mean().sort_values(by='ratings_count', ascending=False)
#Authors with average ratings count who published book before 2003
old_author = dd[['authors','date','ratings_count']][dd['date'] == 'old']
old_author_ratings = old_author[['authors','ratings_count']].groupby('authors').mean().sort_values(by='ratings_count', ascending=False)

mac_data = dd[['authors','average_rating','page_number']].groupby(['authors']).mean()
mac_data['new_ratings'] = new_author_ratings['ratings_count']
mac_data['old_ratings'] = old_author_ratings['ratings_count']
#average ratings label
mac_data['favourite'] = pd.cut(mac_data['average_rating'], bins=[0,4,5], labels=['no','yes'], right=True, include_lowest=True)
mac_data.drop(['average_rating'], axis=1, inplace=True)
# fill NaN with zeros
mac_data[['new_ratings','old_ratings']] = mac_data[['new_ratings','old_ratings']].fillna(0)
#comparison the number of new or old books
for i in mac_data.index:
    if mac_data.loc[i,'new_ratings'] >= mac_data.loc[i,'old_ratings']:
        mac_data.loc[i,'old_ratings'] = 'old'
        mac_data.loc[i,'new_ratings'] = 'new'
    else:
        mac_data.loc[i,'new_ratings'] = 'old'
        mac_data.loc[i,'old_ratings'] = 'new'
mac_data.rename({'new_ratings': 'new_book_ratings','old_ratings': 'old_book_ratings'},axis=1,inplace=True)
mac_data.reset_index(inplace=True)
mac_data.drop(['authors'], axis=1, inplace=True)
# Figure distribution of favourite book between the number of page and books ratings count

# sns.set_context('paper')
sns.color_palette('bright')
sns.catplot(x='new_book_ratings', y='page_number', hue='favourite',
            data=mac_data, kind='swarm', palette=dict(yes='salmon',  no='indigo'))
plt.xlabel('Books Ratings Count')
plt.ylabel('The Number of Page')
plt.title('Favorite Book Distribution in Features')
plt.tight_layout()
plt.savefig(f'plots/favourite_book_distribution.png')
plt.show()
plt.clf()
# Convert categorical variable into dummy variables
m_data = pd.get_dummies(mac_data, columns=['new_book_ratings','old_book_ratings','favourite'], drop_first=True)
m_data.rename({'new_book_ratings_old': 'newbook_ratings_count','old_book_ratings_old': 'oldbook_ratings_count','favourite_yes': 'favourite'},axis=1,inplace=True)
corr = m_data.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
cmap = sns.diverging_palette(240,10, as_cmap=True)
a = sns.heatmap(corr, mask=mask, cmap=cmap, center=0,square=True, linewidths=.2,annot=True)
a.set_title('Correlation Between Features')
# a.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig(f'plots/correlation_heatmap.png')
plt.show()
plt.clf()
#Machine Learning Process
Y = m_data['favourite']
X = m_data.drop(['favourite'],axis=1)
scaler = StandardScaler()
Xn = scaler.fit(X).transform(X)
X = pd.DataFrame(Xn, columns=X.columns)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=50)
model = AdaBoostClassifier(n_estimators = 100, random_state=100, learning_rate=1.3)
model.fit(X_train,Y_train)
predicted = model.predict(X_test)
ac = accuracy_score(Y_test, predicted)
cm = confusion_matrix(Y_test, predicted)
sns.heatmap(cm, annot=True, cmap='GnBu', fmt='g').set_title('Confusion Matrix')
plt.savefig(f'plots/confusionmatrix_heatmap.png')
plt.show()
plt.clf()
