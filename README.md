# pyMAST

Pretty simple, just call in the HurdleLogNormal class. It runs like any scikit-learn regression. 

model = HurdleLogNormal().fit(X, y)


__Access the logistic coefficients__
model.logistic.coef_
model.logistic.intercept_

__Access the regression coefficients__
model.linear.named_step["regressor"].coef_
model.linear.named_step["regressor"].intercept_

As a note, I used Ridge regression to help with cases where there was weird numerical instability (i think?). 
Working on creating utility functions that make it easier to access the coefficients / logFC for instances where people might use this as a drop-in for MAST. 