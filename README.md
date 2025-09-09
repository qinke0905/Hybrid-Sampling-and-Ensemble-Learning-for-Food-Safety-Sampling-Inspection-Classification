# LOF-KNN-CSENN
## Requirements

1. Install Python >= 3.10.

2. Required dependencies can be installed by: 
   
   ```
   pip install -r requirements.txt
   ```

### simulated_imbalanced_data_sampling.py 
This code focuses on comparing and visualizing sampling methods for imbalanced data, with its core purpose being to validate the LOF-KNN-CSENN method against classical sampling methods using a simulated dataset.


#### main.py 
This code integrates the LOF-KNN-CSENN method with the stacking ensemble model, yielding a comprehensive solution that unifies data preprocessing and ensemble learning-based classification. It is specifically tailored for classification tasks involving imbalanced data.

This code requires the input data to be in a structured tabular format, where each row corresponds to one sample and each column represents either a feature or a label. Specifically, the last column of the table serves as the target variable to be predicted, while all columns except the last one are treated as input features for the model.
