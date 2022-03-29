PROJECT DETAILS FOR ZILLOW DATASET

1. Overview Project Goals

    - Construct a machine learning regression model that predicts propert tax assessed values for Single Family Properties.
    - Find key drivers of property value for single family properties.

2. Project Description
    - Zillow is looking to improve their current model for predicting single family property tax values. I have been tasked with providing insights on new, potentially unseen, features that can improve model performance.

3. Initial Hypothesis/Questions

    - Does location affect tax value?
    - does square footage affect tax value?
    - Does bedroom count affect tax value?
    - Does bathroom count affect tax value?

4. Data Dictionary 
   |Column | Description | Dtype|
    |--------- | --------- | ----------- |
    bedroom | the number of bedrooms | int64 |
    bathroom | the number of bathrooms | int64 |
    square_ft | square footage of property | int64 |
    lot_size | square footage of lot | int 64 |
    tax_value | property tax value dollar amount | int 64 |
    year_built | year the property was built | int64 |
    fips | geo code of property | int64 |
    county | county the property is in | object |
    age | the difference between year_built and '2017 | int 64

5. Project Planning

Recreate the plan by following these steps
    
Planning 
    - Define goals
    - What is my MVP?
    - Ask questions / formulate hypotheses
    - Determine audience and deliver format

Acquisition

- Create a function that establishes connection to the zillow_db
- Create a function that holds your SQL query and reads the results
- Create a function for caching data and stores as .csv for ease
- Create and save in acquire.py so functions can be imports
- Test functions

Preparation
- Create a function thatpreps the acquired data
- This function will:
    - remove duplicates
    - remove empty values
    - convert data types
    - encode categorical columns
    - renames columns
    - creates a columns for house 'age'
- Create a function that splits the data into 3 sets. train, validate, test
    - Split 20% (test data), 24% (validate data), and 56%(test data)
- Create functions and save in prepare.py to be easily imported
- Test functions

Exploration 
- Use initial questions to guide the exploration process
- Create visualizations of data
- Statistical analysis to confirm or deny hypothesis
- Save work with notations in zillow_work.ipynb
- Document answers to questions as Takeaways

Model

Delivery
- Report is saved in Jupyter Notebook
-  Presented via Zoom
- The audience is the data science team at Zillow

6. Recreation of Project:
-You will need an env file with database credentials saved to your working directory
    - database credentials (username, password, hostname)
- Create a gitignore with env file inside to prevent sharing of credentials
- Download the acquire.py and prepare.py files to working directory
- Create a final notebook to your working directory
- Review this README.md
- Libraries used are pandas, numpy, matplotlib, Scipy, sklearn, seaborn
- Run zillow_final.ipynb

7. Key Findings and takeaways
- PENDING