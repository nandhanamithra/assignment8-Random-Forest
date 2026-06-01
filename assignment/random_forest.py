import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#data read
data= pd.read_csv(r"C:\flutter_projects\INTERNSHIP\class6\assignment\Iris.csv")

x=data.drop (columns=["Id","Species"])
y=data["Species"]

#databtrain and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42)

model.fit(x_train,y_train)

#prediction and accuracy
y_predict=model.predict(x_test)
accuracy=accuracy_score(y_test,y_predict)
print("ACCURACY",accuracy*100,"%")

flowers = [[5.0, 3.6, 1.4, 0.2], [6.1, 2.8, 4.0, 1.3], [6.9, 3.1, 5.4, 2.1]]
for flower in flowers:
    print("Predicted Species:", model.predict([flower])[0])
    print("Sepal Length=", flower[0])
    print("Sepal Width=", flower[1])
    print("Petal Length=", flower[2])
    print("Petal Width=", flower[3])
