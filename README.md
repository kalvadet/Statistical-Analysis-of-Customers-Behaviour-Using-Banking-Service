# Statistical-Analysis-of-Customers-Behaviour-Using-Banking-Service
Overview
This Python script is designed for performing data analysis on banking customer datasets. It includes operations such as reading CSV files, data cleaning, manipulation, and visualization to understand customer behavior and characteristics.

Features
Data Loading: Load customer data from a CSV file for analysis.
Data Preprocessing: Replace numerical codes in 'IsActiveMember' and 'HasCreditCard' columns with meaningful text labels.
Data Analysis: Calculate the proportion of active vs. non-active members, the status of credit card ownership, and perform a chi-square test to explore the relationship between customer complaints and customer exits.
Data Visualization: Visualize the distribution of active members, credit card ownership, gender, age, tenure, credit score, estimated salary, and location using bar plots and histograms.
Statistical Tests: Perform chi-square tests to examine the relationship between complaints received and customer exits.
Predictive Modeling: Utilize logistic regression to model the likelihood of customer exits based on satisfaction scores.
Installation
Ensure you have Python installed on your machine.

Install required libraries using pip:

Copy code
pip install pandas seaborn matplotlib scipy sklearn statsmodels
Usage
Replace 'C:\\Users\\user\\OneDrive\\Documents\\analysis1\\Main Sample.csv' with the path to your dataset CSV file in the script. Then, run the script in your Python environment to perform the analysis and generate the visualizations.

Contributing
Contributions are welcome! Please feel free to submit pull requests with improvements or report any issues you encounter.

License
This project is open source and available under the MIT License.
