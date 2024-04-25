# All required libraries are imported here for you.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Write your code here
best_predictive_feature={}
crops.head()
crops_features=crops.drop('crop',axis=1)
for feature in crops_features.columns:
    X=crops[[feature]].values
    y=crops['crop'].values
    X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,
                                                             random_state=42)
    
    logreg=LogisticRegression()
    logreg.fit(X_train,y_train)
    best_predictive_feature[feature]=logreg.score(X_test,y_test)
best_feature, best_score = max(best_predictive_feature.items(), key=lambda x: x[1])

print("Best Predictive Feature:", best_feature)
print("Accuracy Score:", best_score)
