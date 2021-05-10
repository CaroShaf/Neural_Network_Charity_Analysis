# Neural Network Charity Analysis
With a database of over 34,000 charities, Alphabet Soup, a charitable organization, wanted to know if applicants for funds were likely to be able to succeed with the requested funds or not.  This is a binary classification problem with many features, so machine learning seemed like a good possibility for determining these probabilities.

Resources: CSV file provided by Alphabet Soup, Python, Pandas, Scikit Learn, TensorFlow, Keras

## Overview of the loan prediction risk analysis
To determine whether or not an applicant would succeed or not, a Sequential model from the Keras API run through the TensorFlow library was studied.  Attempts to optimize the deep learning model were made by adjusting different parameters to see how the accuracy of the predictions was affected.

## Results
### Data Preprocessing
  * What variable(s) are considered the target(s) for your model?
      "IS_SUCCESSFUL" is considered the target for our predictions in this model.
      
  * What variable(s) are considered to be the features for your model?
      Several categorical features were present in the dataset which were bucketed, and some that were left as is.  They included: 
        APPLICATION_TYPE           
        AFFILIATION                
        CLASSIFICATION             
        USE_CASE                    
        ORGANIZATION                
        STATUS                      
        INCOME_AMT                  
        SPECIAL_CONSIDERATIONS      
        ASK_AMT                  
        
  * What variable(s) are neither targets nor features, and should be removed from the input data?
        STATUS was dropped due to the majority being the same value, as as such not contributing any significant information.  Furthermore SPECIAL_CONSIDERATIONS was also
        dropped since the majority didn't require any special considerations.  EIN and Name were dropped since they contributed nothing towards answering the question at hand.


  
### Compiling, Training, and Evaluating the Model
  * How many neurons, layers, and activation functions did you select for your neural network model, and why?
      Because there were approximately 40 features, a 24-neuron input/12-neuron single-hidden layer/single output model was selected.  Althought it was not considered deep, it
      was a starting point.  Eventually, another hidden layer was added.  The activation of all input/hidden layers were ReLu but then modified to SoftMax, with the output layer 
      being Sigmoid always.  These are typical choices for a binary classification problem.

  * Were you able to achieve the target model performance?
      This model did not achieve its target performance exceeding 75%.  All attempts performed at slightly less than 75%
      
  * What steps did you take to try and increase model performance?
      Several attempts were made to adjust the accuracy score by adding an additional hidden layer, changing the scaler from StandardScaler to MinMaxScaler, reducing the number
      of features by elmininating other non-useful columns, as well as changing the activation.


## Summary
None of the attempted adjustments had a significant effect on the performance of this model.  A RandomForest was run, also, just for a comparison but it's accuracy score was
around 70%

## Recommendation
It is difficult to recommend this model, along with any of its adjustments, since the accuracy fairly low.  Since the expenditure of funds to these many organizations is quite 
substantial, it is important to not lose money by making poor predictions of success.  A loss of thousands of dollars annually could have negative implications for continued
success of Alphabet Soup if not careful.  However, the RandomForest model didn't outperforman the Deep Learning Neural Network, so perhaps other simpler models should be 
attempted.  It would also be useful to look at Confusion Matrices for each attempted optimization.  Depending on what level of comfort Alphabet Soup has, the most important
metric could be precision, rather than accuracy score.
