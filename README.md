# Python code for detecting curve's change


## data

Simulated data :
 1. decreasing exponential curve followed by
 2. decreasing linear line
 3. white noise added

## algorithm 

1. slice the complete curve with a moving window (size and lag can be 
adjusted)
2. for each slice
    1. fit a linear model on the slice
    2. compute RMSE between fitted values and true values
    3. append rmse score
3. check when there is a "brutal" change in rmse
    1. visual evaluation
    2. score diff can be used to return the shifting point
