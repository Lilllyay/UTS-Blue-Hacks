import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

url = "https://raw.githubusercontent.com/Lilllyay/Hackthon/codespace-glorious-fishstick-v6v775wxgv7g2xg4j/datasetTrain%20-%20Sheet1.csv"

df = pd.read_csv(url)

print(df.head())

X = df[["Age","Ethnicities","Sex"]]
y = df["Median Weekly Earning"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=50)
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

pickle.dump(classifier,open("model.pkl","wb"))