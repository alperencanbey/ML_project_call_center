# Response Prediction for a Call Center


work-in-progress

Project Summary:

Project Steps:
- Separating data into training and test sets
- Deciding on the variables that explain more than 99% of the variation through calculating the eigenvalues.
- Re-balancing the sample to avoid any biases. Specifically balancing the numbers of successfull and unsuccessfull calls.
- Scaling the data
- 5x2 cross vaildation
- Applied different methods such as Naive Bayes, MLP with different sizes, and Decision Tree.
- Choosed the method that gives highest AUROC, and applied it to predict the test data. 
