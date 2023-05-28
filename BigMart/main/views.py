from django.shortcuts import render
from django.http import JsonResponse  
from json import *
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rest_framework.views import APIView



class GetData(APIView):  
    def post(self, request,format=None):  

        #  Read the training data.csv file and the request's body data
        df = pd.read_csv('main/Train.csv')
        body = request.data

        # Convert the data of the body to a proper form
        d = {'Item_Identifier': [body['Item_Identifier']], 'Item_Weight': [float(body['Item_Weight'])] ,'Item_Fat_Content': [body['Item_Fat_Content']], 'Item_Visibility': [float(body['Item_Visibility'])],'Item_Type': [body['Item_Type']], 'Item_MRP': [float(body['Item_MRP'])],
            'Outlet_Identifier': [body['Outlet_Identifier']],'Outlet_Establishment_Year': [int(body['Outlet_Establishment_Year'])],'Outlet_Size': [body['Outlet_Size']],'Outlet_Location_Type': [body['Outlet_Location_Type']],'Outlet_Type': [body['Outlet_Type']]
        }


        # insert the data to the dataframe
        inputData = pd.DataFrame(data=d)

        # Fixing data and missing values
        df = pd.concat([df,inputData])
        # item weight mean
        df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
        # outlet size mode
        df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
        # IF item visibility is zero we set the mean value to them
        df["Item_Visibility"][df["Item_Visibility"]==0]=df['Item_Visibility'].mean()
        # Calculate the establishment years from our current year
        df['Outlet_Years'] = 2023 - df['Outlet_Establishment_Year']

        # Replace irregular data
        df.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)

        # Normalize the data and convert it to a numerical form
        encoder = LabelEncoder()
        df['Item_Identifier'] = encoder.fit_transform(df['Item_Identifier'])
        df['Item_Fat_Content'] = encoder.fit_transform(df['Item_Fat_Content'])
        df['Item_Type'] = encoder.fit_transform(df['Item_Type'])
        df['Outlet_Identifier'] = encoder.fit_transform(df['Outlet_Identifier'])
        df['Outlet_Size'] = encoder.fit_transform(df['Outlet_Size'])
        df['Outlet_Location_Type'] = encoder.fit_transform(df['Outlet_Location_Type'])
        df['Outlet_Type'] = encoder.fit_transform(df['Outlet_Type'])

        # Removing the target feature from dataset to predict data and the query that was just added
        # Test is the query data we just sent
        test=df[-1:]
        df=df[:8523]
        test.drop(['Item_Outlet_Sales'],inplace=True, axis=1)
        #Let's have all the features in X & target in Y
        X = df.drop(columns='Item_Outlet_Sales', axis=1)
        y = df['Item_Outlet_Sales']

        # Train data and split it 30% test and training 70%
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)

        # StandardScaler removes the mean and scales each feature/variable to unit variance. 
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        test=sc.transform(test)

        # Based on our comparisons we choose xgb model due to its higher accuracy
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)

        
        # Prediction of sales with the model.predict using the test data
        sales_data_prediction = model.predict(test)       
        prediction = str(sales_data_prediction[0])
        
        return JsonResponse({"prediction":prediction})