
                                 <h1>BUILDING A MEDICAL INSURANCE COST PREDICTION MACHINE LEARNING MODEL</h1>

AIM:

   1) To predict medical costs for patients based on demographic factors,
       lifestyle-related information, health status, insurance plan type,
      family medical history, distance to the nearest hospital, primary care physician visits and other features.

   2) To provide actionable insights for Hospital ABC to improve resource
      allocation, healthcare planning, and patient support services, ultimately enhancing the quality and accessibility of healthcare services for its patients.



PROJECT DESCRIPTION:

Hospital ABC, a leading healthcare provider, aims to optimize its resource allocation and improve
patient care by understanding the factors influencing medical costs for its patients. To achieve
this, Hospital ABC has collected a comprehensive dataset containing information about patients'
demographics, lifestyle, health status, and medical expenses etc. The dataset includes attributes
such as age, gender, BMI, smoking status, region, as well as additional information like
occupation, exercise frequency, chronic conditions, and more.The project aims to deliver a predictive model that can accurately estimate medical costs for
patients based on their demographic, lifestyle, health status, and other relevant factors.
Additionally, the project will provide actionable insights for Hospital ABC to improve resource
allocation, healthcare planning, and patient support services, ultimately enhancing the quality and
accessibility of healthcare services for its patients.



 DATA SOURCE:
  
  https://drive.google.com/drive/recent
 
 DATASET DESCRIPTION:

 This dataset contains 1000 rows and 35 columns.Columns are Age, BMI, Children, Chronic_Conditions,
Distance_to_Nearest_Hospital, Family_Medical_History,
Primary_Care_Physician_Visits, Emergency_Room_Visits,
Healthcare_Utilization, Charges,Sex, Smoker, Region, Occupation,Exercise_Frequency,Insurance_Plan_Type, Marital_Status,Income_Level,Mental_Health_Status, Prescription_Medication_Usage,Employment_Status, Education_Level,Dietary_Habits,Alcohol_Consumption, Sleep_Quality,Stress_Level,Medication_Adherence, Physical_Activity, Access_to_Healthcare,Social_Support,Environmental_FactorsGenetic_Predisposition,Health_Literacy_Level, Comorbidities, Access_to_Telemedicine.

In this case Charges is the Target column,other columns are features(columns that are used to predict the values of target)

This dataset contains  null values in all columns, it's crucial to preprocess the data before training a machine learning model. 
 
 TECHNOLOGIES USED

 1)Python-programming language for development.

 2)Power bi-For data visualisation

 3)Seaborn-Data Visualisation

 4)Matplotlib-Data Visualisation

 5)Scikit-learn- a versatile machine learning library in Python, provided essential functionalities for tasks such as data preprocessing, feature selection, and building machine learning models.

 
 
STEPS INVOLVED:

1)Dataset loading using read_csv from pandas library.

 df=pd.read_excel("medical_new.excel"),where df is our dataframe

2)To find  the first 5 rows of our Dataset: df.head() can be used.To find the last 5 rows of our dataset :df.tail() can be used.

3) To find the number of non-null columns and the datatypes of each column ,use df.info()

4)To find the statistical summary of our numerical columns use df.describe()

DATA PREPROCESSING

1)Checking for duplicates in our dataset using duplicated() .No duplicates in our datset.

2)Checking for null values :df.isna().sum().There are null values in all the columns.

3)Imputing or replacing null values in the numerical columns using mean of each column. Imputing with the mean helps maintain the overall distribution of the data.

4) Checking for outliers in our dataset.

Outliers are data points that significantly differ from other observations in a dataset. They are extreme values that lie outside the typical range of the majority of the data

5) Inorder to identify outliers I have plotted BOX PLOT.

A boxplot, also known as a box-and-whisker plot, is a graphical representation of the distribution of a dataset through five summary statistics: the minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum.

Here's a breakdown of its components:

Box: The central rectangle spans from the first quartile (Q1) to the third quartile (Q3). This region represents the middle 50% of the data, known as the interquartile range (IQR). The line inside the box represents the median.

Whiskers: Lines (or sometimes bars) extend from the edges of the box to the minimum and maximum values within a certain range from the quartiles. The range can be defined in various ways, such as 1.5 times the IQR, or it may extend to the actual minimum and maximum values in the dataset.

There are outliers in BMI and HealthCare_Utilization columns.


6)To find the outlier values for further processing,I have used IQR(interquartile range).

The interquartile range (IQR) is a measure of statistical dispersion that represents the spread of the middle 50% of the data. It is calculated as the difference between the third quartile (Q3) and the first quartile (Q1) of a dataset.

Here's how to calculate the IQR:

Sort the Data: First, sort the dataset in ascending order.

Find the Quartiles: Divide the sorted data into four equal parts, or quartiles. The first quartile (Q1) is the median of the lower half of the dataset, and the third quartile (Q3) is the median of the upper half of the dataset.

Calculate the IQR: Subtract Q1 from Q3 to obtain the interquartile range (IQR). Mathematically, IQR = Q3 - Q1.

The IQR is a robust measure of variability that is less sensitive to outliers compared to other measures such as the range (difference between the maximum and minimum values) or the standard deviation. It focuses solely on the middle 50% of the data, making it a useful tool for describing the dispersion of a dataset while mitigating the influence of extreme values.


I have defined  a function to find outliers in each column using IQR and that function returns outlier values in each column.

7) Iam not dealing with Outliers in HealthCare_Utilization,because I think it's normal to have such values in that column.

8)There are outliers in BMI column ,so I just replaced values below 13 and values above 37 in BMI column with the median value in BMI column.

The median value is chosen as a robust measure of central tendency that is less sensitive to extreme values compared to the mean.

Using the median helps mitigate the influence of outliers while preserving the overall distribution and characteristics of the BMI data.

9) Next step is dealing with null values in categorical columns.Just replaced null values n categorical columns with mode values of each categorical columns.


10)Plot scatterplot to find relationship between Age,Distance_to_NearestHospital,BMI and Charges.
 i)There is a strong positive correlation between Age and Charges.
 ii)There is a positive correlation between BMI and Charges.
 iii)There is no correlation beteen Distance_to_NearestHospital and Charges.

11)plotted Charges by Sex and Smoker in powerbi using barchart:
   
   i)Most of our patients are Male.

   ii)Higher charges for smokers: The graph likely  shows that smokers tend to have higher total charges compared to non-smokers.

12) plotted Charges by Region and Insurance_plan type using barchart.

   i)It appears most patients are from the East region, Among all the regions, patients in the East seem to have the most charges overall.

 ii)Bronze appears to be the most common insurance plan type used by patients across all regions.

13)plotted Charges by Excerscise frequency  and Dietry_Habits type using barchart.
  
  i)For all exercise frequencies, healthy patients tend to have significantly lower total charges compared to unhealthy patients. This is likely because unhealthy patients require more medical care and treatment.

14)plotted Charges by Mental Health using barchart.

   i)In general, patients with Fairer mental health seem to have higher total charges.

15) Health literacy in diffrent regions using count plot .

   i)A large number of patients across all regions lack health literacy.

 ii)Hospital ABC can Implement targeted health education programs.These programs can focus on improving health literacy by providing easily understandable information about common health conditions, preventive measures, and available healthcare services.

16)healthcare Acess in diffrent regions using count plot.

  i)Access to Healthcare is difficult across different regions.

 ii)Hospital ABC can  Implement mobile healthcare services to reach underserved areas with limited access to healthcare facilities. Mobile clinics equipped with medical staff, diagnostic equipment, and basic treatment capabilities can provide essential healthcare services directly to communities in need.

16)Plotted Charges vs Chronic conditions uing piechart.

 i)Most of the patients have no Chronic conditions,some of them have one chronic condition and some of them have more than one chronc condition.

 ii)Couldnot find any correlation between Charges and Chronic_Conditions.
 

17) Next we need to encode this categorical columns that means our machine learning model only knows numerical data ,it cannot process categorical data,for that I choose label encoding.

Label encoding is a process of assigning numerical labels to categorical data values. It is a simple and efficient way to convert categorical data into numerical data that can be used for analysis and modelling.

The basic idea of label encoding is to assign a unique integer to each category in a categorical variable. For example, if we have a categorical variable “colour” with categories “red”, “green”, and “blue”, we can assign the labels 0, 1, and 2 respectively. This allows us to represent the data numerically, which is necessary for many machine learning algorithms.

18) Next step is Feature scaling our features.
The goal of feature scaling is to ensure that all features contribute equally to the analysis and model training process, preventing features with larger scales from dominating those with smaller scales.

I have used Standard Scaler,In standardization, also known as Z-score normalization, each feature is transformed to have a mean of 0 and a standard deviation of 1.
 
Standardization preserves the shape of the distribution of each feature and is less affected by outliers compared to min-max scaling.


19)Finding correlation matrix.

Correlation matrix is used to find the linear correlation between features against features and also features against target.

It is calculated using corr().
1 for strong positive correlation.
0 for zero correaltion.
-1 for negative correaltion.


20)Feature Selection using Random Forest regressor to improve my model performance.


 MACHINE LEARNING

Machine learning :Machine learning is a subset of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computers to perform tasks without explicit programming instructions. In essence, it's about teaching machines to learn from data in order to make predictions or decisions.

1)Splitting data into feature set and target.And also splitting dataset into training and testing dataset in 80:20 ratio.By dividing the dataset into training and testing we can evaluate our model using testing dataset ,after training it with the training dataset.

2)Selecting models like Linear regression,Ensemble methods like Random Forest Regressor,XGBoost regressor,Adaboost ,Decision Tree Regressor for model training.




MODELS USED:

Linear Regression: 

       Linear regression is a fundamental statistical method used for modeling the relationship between a dependent variable (target) and one or more independent variables (features). It assumes a linear relationship between the independent variables and the dependent variable.

Decision Tree Regressor:   

A Decision Tree Regressor is a type of supervised learning algorithm used for regression tasks. It builds a predictive model in the form of a tree structure where each internal node represents a decision based on the value of a feature, each branch represents the outcome of the decision, and each leaf node represents the predicted continuous value.

Here's how a Decision Tree Regressor works:

Splitting Data: The algorithm begins by selecting the feature and the split point that best divides the dataset into subsets that minimize the variance of the target variable (or maximize another criterion, such as minimizing the mean squared error).

Recursive Partitioning: The dataset is recursively partitioned into subsets based on the selected split point. This process continues until a stopping criterion is met, such as reaching a maximum tree depth, having a minimum number of samples in each leaf node, or no further improvement in the split criterion.

Leaf Node Prediction: Once the tree is fully grown, each leaf node contains a prediction for the target variable. This prediction is typically the mean (or median) of the target variable values in the leaf node.

Tree Pruning: After growing the tree, pruning techniques may be applied to reduce its size and complexity, thereby improving generalization performance and preventing overfitting.

Prediction: To make predictions for new instances, the Decision Tree Regressor traverses the tree from the root node to a leaf node based on the feature values of the instance. The predicted value is then obtained from the leaf node corresponding to the path taken.

Decision Tree Regressors offer several advantages:

Interpretability: Decision trees are easy to interpret and visualize, making them useful for gaining insights into the relationships between features and the target variable.
Non-linearity: Decision trees can capture non-linear relationships between features and the target variable, making them suitable for modeling complex datasets.
Robustness to Irrelevant Features: Decision trees are robust to irrelevant features and can handle datasets with a mixture of numerical and categorical features.

Ensemble Methods:
        Ensemble methods are machine learning techniques that combine the predictions of multiple individual models to produce a stronger, more robust predictive model. These methods leverage the concept of "wisdom of the crowd," where aggregating the predictions of multiple models can often outperform any single model.

There are several types of ensemble methods, but two of the most popular are:

Bagging (Bootstrap Aggregating):

Bagging involves training multiple instances of the same base learning algorithm on different subsets of the training data, typically obtained through bootstrapping (random sampling with replacement).
Each model in the ensemble learns from a slightly different perspective of the data, reducing variance and improving generalization performance.
The final prediction is obtained by averaging (for regression) or voting (for classification) the predictions of all models.

Boosting:

Boosting works by training a sequence of weak learners (models that perform slightly better than random chance) sequentially, where each subsequent model focuses on the mistakes made by the previous ones.
In each iteration, the algorithm assigns higher weights to instances that were misclassified by earlier models, thereby emphasizing the difficult-to-predict cases.
The final prediction is a weighted combination of the predictions made by all models in the sequence.
Ensemble methods offer several advantages, including:

Improved predictive performance: Ensemble methods often yield better generalization performance compared to individual models, especially when the base models are diverse and complementary.
Robustness to overfitting: By combining the predictions of multiple models, ensemble methods can mitigate overfitting and reduce variance, leading to more stable and reliable predictions.
Flexibility and scalability: Ensemble methods can be applied to a wide range of machine learning algorithms and are easily scalable to large datasets.

i)Random Forest Regressor:
            Random Forest: The Random Forest algorithm consists of a collection of decision trees. Each decision tree is trained independently on a random subset of the training data, and each split in the tree is based on a randomly selected subset of features.

Bootstrap Aggregation (Bagging): Random Forest employs a technique called bootstrap aggregation, or bagging, to train multiple decision trees. This involves randomly sampling the training data with replacement to create multiple bootstrap samples, and then training a decision tree on each sample.

Decision Trees: Each decision tree in the Random Forest is trained to predict the target variable based on the values of the input features. The trees are grown recursively by splitting the data at each node based on the feature that best separates the data according to a specified criterion, such as minimizing the mean squared error.

Ensemble Prediction: Once all the decision trees are trained, predictions are made by aggregating the predictions of individual trees. For regression tasks, the final prediction is typically the mean or median of the predictions made by all trees in the forest.

The Random Forest Regressor offers several advantages:

High Performance: Random Forests tend to provide high predictive performance, often outperforming single decision trees and other machine learning algorithms.
Robustness: Random Forests are less prone to overfitting compared to individual decision trees, thanks to the ensemble approach and the randomness introduced during training.
Feature Importance: Random Forests can provide insights into feature importance, helping to identify the most relevant features for predicting the target variable.


ii)XGBoost:

   XGBoost (Extreme Gradient Boosting) Regressor is a powerful and popular machine learning algorithm used for regression tasks. It is an implementation of gradient boosting, a machine learning technique that builds an ensemble of weak learners (decision trees) sequentially to make accurate predictions.

Here's how the XGBoost Regressor works:

Gradient Boosting: XGBoost follows the principle of gradient boosting, where weak learners (decision trees) are sequentially added to the ensemble, with each tree trained to correct the errors made by the previous ones.

Objective Function: XGBoost optimizes a user-specified objective function that measures the difference between the predicted and actual values. The objective function typically consists of two parts: a loss function that quantifies the difference between the predicted and actual values, and a regularization term that penalizes model complexity to prevent overfitting.

Boosting Iterations: During training, XGBoost adds decision trees to the ensemble iteratively. In each iteration, the algorithm fits a new decision tree to the residuals (the differences between the predicted and actual values) of the previous predictions.

Tree Construction: XGBoost constructs decision trees greedily, by recursively partitioning the data into regions that minimize the loss function. It uses a technique called gradient boosting to optimize the split points, considering both the prediction errors and the model's complexity.

Regularization: XGBoost includes several regularization techniques to control overfitting and improve generalization performance, such as shrinkage (learning rate), tree depth regularization, and column subsampling.

Prediction: Once training is complete, XGBoost makes predictions by aggregating the predictions of all the trees in the ensemble. The final prediction is the sum of the predictions made by each tree, weighted by a learning rate.

XGBoost Regressor offers several advantages:

High Performance: XGBoost is known for its high predictive accuracy and efficiency, often outperforming other machine learning algorithms on a wide range of datasets.
Flexibility: XGBoost can handle both numerical and categorical features, and it supports various objective functions and evaluation metrics, making it versatile for different types of regression problems.
Robustness: XGBoost includes regularization techniques to control overfitting and handle noisy data, making it robust and less prone to overfitting compared to simpler models.



iii)Adaboost Regressor:

   AdaBoost Regressor is a variant of the AdaBoost algorithm specifically designed for regression tasks. While AdaBoost is commonly associated with classification, AdaBoost Regressor applies the same principles of boosting to regression problems, aiming to predict continuous numeric values rather than discrete class labels.

Here's how AdaBoost Regressor works:

Base Learner Initialization: AdaBoost Regressor begins by fitting a base regression model to the entire training dataset. The base learner is typically a weak regression model, such as a decision tree with limited depth or a simple linear regression model.

Weighting Data: Initially, each training example is assigned an equal weight.

Iterative Training: AdaBoost Regressor iteratively trains multiple instances of the base regression model. In each iteration:

The algorithm fits a new weak regression model to the training data, with higher emphasis on examples that were poorly predicted by the previous models.
The weights of training examples are updated based on the errors of the previous models. Examples with larger errors receive higher weights to prioritize their correct prediction in subsequent iterations.
Each weak regression model focuses on minimizing the errors made by the ensemble of previous models.
Combining Predictions: After training all weak regression models, AdaBoost Regressor combines their predictions by averaging their outputs or using weighted averaging, where models with better performance are given higher weights.

Final Prediction: The final prediction of the AdaBoost Regressor is obtained by aggregating the predictions of all weak regression models.

AdaBoost Regressor offers several benefits:

High Predictive Performance: By combining the predictions of multiple weak regression models, AdaBoost Regressor often achieves high accuracy in predicting continuous target variables.
Robustness: AdaBoost Regressor is less prone to overfitting compared to individual weak models, as it focuses on minimizing errors across multiple models.
Flexibility: AdaBoost Regressor can be used with various base regression models, allowing flexibility in model selection based on the characteristics of the dataset.
 METRICS USED:

 1)Mean Absolute Error:
     
     In machine learning, the Mean Absolute Error (MAE) is a common metric used to evaluate the performance of regression models. It measures the average absolute difference between the predicted values and the actual values in the dataset.

Lower values of MAE indicate better model performance, with a MAE of 0 indicating perfect predictions where the predicted values exactly match the actual values.


2)Rsquared(coefficient of determination):
  
  The coefficient of determination, R2
 , is a statistical measure that represents the proportion of the variance in the dependent variable (target) that is predictable from the independent variables (features) in a regression model. In other words, it quantifies the goodness of fit of the model to the observed data.
 


 SELECTED MODEL

 I have choosen Random Forest Regressor,beacause it gives me low MAE and a good R2score.


Then plotted actual vs predicted values for a random regressor model.
 SAVING MY MODEL:

 joblib is a Python library that provides utilities for saving and loading machine learning model.

 import joblib
joblib.dump(rf_regressor, 'model_filename.pkl')
## rf_regressor is my mdel,model_filename is the file to which my model is saved.
 CONCLUSIONS:


1) Employed linear regression alongside ensemble methods such as Random Forest Regressor, XGBoost, AdaBoost, and Decision Trees due to the limited linear relationship observed in the data. Notably, Random Forest Regressor yielded a commendable R2 score while minimizing mean absolute error.


2)Age and BMI, along with the frequency of Primary Care Physician Visits, exhibit a positive correlation with medical charges.


3)Medical charges tend to be higher for male smokers, who comprise a significant portion of the patient population.


4)Hospital ABC can enhance its patient support services by implementing comprehensive programs specifically designed for male smokers. These initiatives should encompass counseling sessions, structured smoking cessation programs, and the distribution of educational materials focusing on fostering healthier habits and mitigating the risk of chronic diseases.

5)The East region appears to have the highest patient volume, correlating with the region's highest overall charges. Bronze insurance plans emerge as the prevailing choice among patients across all regions.

6)Across all exercise frequencies, healthy patients exhibit notably lower total charges compared to unhealthy patients, presumably due to the increased medical care and treatment required by the latter.

7)Hospital ABC can address the widespread lack of health literacy across all regions by implementing targeted health education programs. These initiatives aim to enhance understanding of common health conditions, preventive measures, and available healthcare services through easily accessible and comprehensible information.

8)Additionally, engaging with local communities through community outreach initiatives is essential. Collaborating with community organizations, hosting health fairs, and conducting workshops can effectively raise awareness about health literacy and encourage healthy behaviors among residents in these regions.

9)Hospital ABC can enhance access to healthcare in underserved regions by introducing mobile healthcare services. These mobile clinics, staffed with medical professionals and equipped with diagnostic tools and basic treatment capabilities, can bridge the gap in areas with limited access to traditional healthcare facilities. This proactive approach ensures essential healthcare services are directly delivered to communities in need, thereby improving overall healthcare accessibility.
