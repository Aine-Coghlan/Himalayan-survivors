# Himalayan-survivors

This is a machine learning project focused on predicting whether a climber survived their expedition or not given a number of factors. The dataset source is The Himalayan Database.

SETUP AND RUN INSTRUCTIONS
Clone this repo to your local device
In terminal --> python -m venv .venv
                .venv\Scripts\activate
                pip install -r requirements.txt
                jupyter notebook
Run 01_data_audit.ipynb and then 02_baseline.ipynb


DATA AUDIT
Irrelevant columns/columns that could result in data leakage have been removed. Some columns, such as mroute1,2+3 and mo2used/mo2none were merged together for better readability and accuracy.

Certain data quality errors were also handled. Such as incorrect data entry in yob and myear resulting in climbers being too old (116), too young (2) or not even born (-51). Any ages that were below 13 (youngest person to attempt to climb) or 80 (oldest person) were removed, to be filled in during preprocessing. Yob column was also removed to avoid contradicting ages.

After removing columns, replacing yob with age etc., There's 89000 rows and 30 columns. Most types are boolean/string (categorical). Two are numerical, age and myear. 5 features are missing values, the highest being mo2note which is an optional category. Age is missing 0.089, citizen 0.000079, residence 0.099, and occupation .34. These will be handled during pre-processing.

There are some duplicates after removing multiple columns, but they represent actual climbers who happen to have the exact same values as the number of duplicates prior to removing the columns is 0.

PRE-PROCESSING
The dataset is split into training and test sets in a stratified 80/20 split. Its stratified because the data's imbalanced and 98/99% of people survived the climb. The test data is hidden from the training data to prevent leakage.

During pre-processing, any missing numerical values were filled in using the median value of the column and categorical ones were just replaced with 'missing'. The median value was of the training set only to prevent leakage. 

EVALUATION DESIGN
In the dummy model, the model basically just chose the most frequent outcome for each climber (they survived). This lead to a 98% accuracy, a 49% F1 Score and an ROC-AUC score of 50. Which essentially means that the model got 98% of outcomes correct which looks good on its own but thats only because 98% of the climbers survived. The F1 Score is because it never guessed death for the climbers that died, so the precision isn't that great. The ROC-AUC score confirms its just guessing and isn't using features to determine the outcome.

In the logistic regression model, the accuracy is a little bit higher, as is the F-Score. The ROC-AUC is 28% higher which is good as it confirms this model is taking the different features into account. These accuracy results should increase with the more advanced models.

ASSISTANCE DECLARATION
I used stack overflow and code included in the lecture slides to help me, particularly when creating the pipelines and models.