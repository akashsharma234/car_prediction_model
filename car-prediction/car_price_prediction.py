from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Load Data
df = pd.read_csv(r"/kaggle/input/ford-car-price-prediction/ford.csv")

# EDA
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()

# Visualization
sns.histplot(df['price'], bins=50, kde=True)
sns.heatmap(df.corr(numeric_only=True), annot=True)

plt.figure(figsize=(10, 8))
sns.boxplot(data=df, x='year', y='price')
plt.xticks(rotation=90)
plt.title("Years vs Price")

sns.scatterplot(data=df, y='mileage', x='price')
plt.title("Mileage vs Price")

sns.lineplot(data=df, x='engineSize', y='price')
plt.xticks(ticks=[i*0.5 for i in range(11)])
plt.title("Engine Size vs Price")

df.columns

sns.boxplot(data=df, x='transmission', y='price')

sns.boxplot(x=df['model'], y=df['price'])
plt.xticks(rotation=90)
plt.title("Model vise prices")

sns.lineplot(data=df, x='model', y='tax')
plt.xticks(rotation=90)
plt.title('Taxes on Models')

X = df.drop(columns=['price'], axis=1)
y = df['price']

X

# Data Preprocessing¶

X = df.drop(columns=['price'], axis=1)
y = df['price']

X

X_one_hot = pd.get_dummies(
    X, columns=['model', 'transmission', 'fuelType'], drop_first=True)
X_one_hot

X_one_hot = X_one_hot.astype(int)
X_one_hot


label_encoder = LabelEncoder()

columns = ['model', 'transmission', 'fuelType']

X_label = X
for i in columns:
    X_label[i] = label_encoder.fit_transform(X_label[i])

X_label

scaler = StandardScaler()

numeric_col = ['year', 'mileage', 'tax', 'mpg']

X_one_hot[numeric_col] = scaler.fit_transform(X_one_hot[numeric_col])
X_one_hot

numeric_col2 = ['year', 'mileage', 'tax', 'mpg', 'model', 'fuelType']

X_label[numeric_col2] = scaler.fit_transform(X_label[numeric_col2])
X_label


# one_hot Encoding
X_train, X_test, y_train, y_test = train_test_split(
    X_one_hot, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

r2_one_hot = r2_score(y_test, y_pred)
r2_one_hot

# label Encoding
X_train, X_test, y_train, y_test = train_test_split(
    X_label, y, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)
y_pred

r2_label = r2_score(y_test, y_pred)
r2_label

# Conclusion
if r2_one_hot > r2_label:
    print("One_hot Encoding Perform better in this model")
else:
    print("label Encoding Perform better in this model")
