# Key Observations From Visuals

    1. The dataset contains 1966 high-risk rows and 1034 low-risk rows, showing that the target is moderately imbalanced toward high-risk outcomes.
    2. The weather categories with the highest high-risk share are: Foggy (69.79%), Hazy (66.61%), Clear (64.81%).
    3. The road types with the highest high-risk share are: State Highway (67.32%), National Highway (66.89%).
    4. High-risk scenarios have a higher average speed limit (75.26 km/h) than low-risk scenarios (74.33 km/h).
    5. Lighting conditions, road condition, and number of vehicles appear as the strongest XGBoost features, indicating that environment and traffic density matter more than road type alone.
    6. The best holdout-accuracy model in the current comparison is Logistic Regression with accuracy 0.6550 and ROC-AUC 0.5417.
