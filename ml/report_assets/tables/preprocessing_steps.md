# Data Preprocessing Steps

        1. Loaded the pedestrian accident dataset from the project dataset folder.
        2. Removed rows with missing values using `dropna()`.
        3. Trimmed column names to prevent whitespace mismatches.
        4. Converted `Time of Day` into a categorical feature called `Time_Category`.
        5. Built the binary target `High_Risk`, where Serious and Fatal cases are mapped to 1.
        6. Dropped non-model columns: `Accident Severity`, `High_Risk`, `Pedestrian_Involved`, and the raw `Time of Day`.
        7. Label-encoded all categorical predictors to make them model-ready.
        8. Split the encoded data into stratified training and testing sets with an 80/20 ratio.
        9. Trained Logistic Regression, Decision Tree, Random Forest, and XGBoost on the same split for comparison.
