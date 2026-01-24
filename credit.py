import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,f1_score,recall_score,precision_score
from imblearn.over_sampling import RandomOverSampler

df=pd.read_csv('credit_card.csv')

print(df.head())
print(df.info())

x=df[["Annual_Fees","Activation_30_Days","Customer_Acq_Cost",
      "current_year","Credit_Limit","Total_Revolving_Bal","Total_Trans_Amt","Total_Trans_Vol","Avg_Utilization_Ratio","Interest_Earned"]]
y=df["Delinquent_Acc"]
print(x.info())

ore=OneHotEncoder(sparse_output=False,drop="first")
le=df[["Card_Category","Week_Num","Use Chip","Qtr","Exp Type"]]
encode=ore.fit_transform(le)
encoded_df=pd.DataFrame(encode,columns=ore.get_feature_names_out(le.columns))

X=pd.concat([x,encoded_df],axis=1)
rus=RandomOverSampler()
rx,ry=rus.fit_resample(X,y)
x_train,x_test,y_train,y_test=train_test_split(rx,ry,test_size=0.2,random_state=42)   

knc=KNeighborsClassifier(n_neighbors=3)
knc.fit(x_train,y_train)

print("Training Accuracy:",knc.score(x_train,y_train))
print("Testing Accuracy:",knc.score(x_test,y_test))
y_pred=knc.predict(x_test)

print(classification_report(y_test,y_pred))
print("F1 Macro:", f1_score(y_test, y_pred, average='macro'))
print("Recall Macro:", recall_score(y_test, y_pred, average='macro'))
print("Precision Macro:", precision_score(y_test, y_pred, average='macro'))

cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


