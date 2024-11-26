# Machine Learning project by Mohammad Imran(2024042724)


#### In this project, I've used *linear Regression Model* to predict the salaries of employees


## At first, Lets scrap the data from LinkedIn.
### Import Dependencies
      from selenium import webdriver
      from selenium.webdriver.common.by import By
      from selenium.webdriver.common.keys import Keys
      from selenium.webdriver.support.ui import WebDriverWait
      from selenium.webdriver.support import expected_conditions as EC
      import requests
      import pandas as pd
      from bs4 import BeautifulSoup
      import time

### Entire python code for scraping -> [link](https://github.com/Imran-2024042724/ML_project/blob/746d1b3961147c21f905b83035be8134e07e51b1/my_linkedin.py)

## Now, Lets build our _Linear Regression_ Model
### Dependencies
  
      import pandas as pd
      from sklearn.model_selection import train_test_split
      from sklearn.linear_model import LinearRegression
      from sklearn.preprocessing import OneHotEncoder
      from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

### Reading the Data
      df = pd.read_csv("jobs.csv")
      df.head()

### converting 'gender' into numerical format

      encoder = OneHotEncoder()
      gender_encoded = encoder.fit_transform(df[['gender']]).toarray()
      gender_columns = encoder.get_feature_names_out(['gender'])
      df[gender_columns] = gender_encoded

### Selecting the Features

      X = df[['years_of _experience'] + list(gender_columns)] 
      y = df['salary'] 

### Split the data into training and test

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### create a LinearRegression Model

      model = LinearRegression()
      model.fit(X_train, y_train)

### Evaluate the Model
      y_pred = model.predict(X_test)
      mae = mean_absolute_error(y_test, y_pred)
      mse = mean_squared_error(y_test, y_pred)
      rmse = root_mean_squared_error(y_test, y_pred)
      
      print("Model Evaluation:")
      print(f"Mean Absolute Error (MAE): {mae}")
      print(f"Mean Squared Error (MSE): {mse}")
      print(f"Root Mean Squared Error (MSE): {rmse}")

### Now, lets make predictions based on a dummy data
      dummy_data = pd.DataFrame({
          'years_of _experience': [5, 10, 3],
          'gender_Female': [1, 0, 0],  
          'gender_Male': [0, 1, 0]     
      })
      predicted_salaries = model.predict(dummy_data)
      print("\nPredicted Salaries for Example Data:")
      print(predicted_salaries)
### Final output
      Predicted Salaries for Dummy Data:
      5 years, female = 64670.47446383 
      10 years,, male = 80373.7660876  
      3 years, male = 58848.8240569 ]

## I've used Tableau public to plot the data

<p align="center">
    <img src="https://github.com/Imran-2024042724/ML_project/blob/main/Dashboard%201.png">
</p>

<div align="center">

# --⏹️ The End ⏹️--

</div>
