# credit_risk_classification
Module_20 Assignment  credit_risk_classification

#### By _**Navyasri Pusuluri**_

## Tools Used

* _jupyter notebook_
* _Git Bash_

## Modules Used

* _pandas_
* _numpy_
* _sklearn.metrics_
* _sklearn.linear_model_
* _imblearn.over_sampling_



## Details of Assigments.

* All the analysis can be found at  main level below path credit_risk_classification.

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
       The purpose of the analyze the dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.
* Explain what financial information the data was on, and what you needed to predict.
        The dataset reflected the following financial information: the loan size, the interest rate, the borrower income, the debt-to-income ratio, the number of accounts, the derogatory marks and the total debt of each loan application.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
         In the dataset, we can see there is a discrepancy between the healthy loans (0) counts and the high-risk loans (1) counts: 75036 healthy loans count vs 2500 high-risk count.
* Describe the stages of the machine learning process you went through as part of this analysis.
        I went through the following stages of the machine learning to process the analysis: Pre-processing (cleaning and preparing the raw data), then training (recognizing the pattern), then validating and evaluating the model.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
        First, I used the supervised learning classification model called Logistic Regression to predict the probability of the binary healthy loans vs high-risk loans event to occur. Then, I used the resampling method to generate more data points as to balance the dataset and make a more accurate model.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
    * Precision- The model performs better healthy loan(precision 1) than high_risk loan (precision 0.87).
    * recall- The model performs better healthy loan (recall 1) than high_risk loan (recall 0.89).
    * Accuracy - The balanced accuracy socre is 94%.
    * confusion_matrix - the true positive value is 18679 and true negative value is 558.
    * support- According to the support column, the model is imbalanced in the training data(healthy loan 18759 vs high_risk loan 625), indicating the structural weakness in the reported scores, in the evaluation process.
   
* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
     * precision - The ratio for both the healthy loan and high_risk loan are equally performs well.
     * recall- The healthy loan and high_risk loan ratios are equal.
     * Accuracy- The accuracy for this model is almost 100%. The true postive and true negative values is also almost equal.
     * Support- According to the support column, the model is balanced in the training data(healthy loan 56277 vs high_risk loan 56277), indicating the structural strength in the reported scores, in the evaluation process.
    
## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
   * The best performance would be the "machine learning model 2", because it reported scores are balanced and shows high levels of Accuracy, Precision and Recall for both healthy loans and high-risk loans.
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
   * With more data points after the resampling, the model 'Machine Learning Model 2' demonstrated higher performance, being more suitable to predict the loan risks and classify them as healthy loans or high-risk loans.
