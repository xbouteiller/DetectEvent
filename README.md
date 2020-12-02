# Python Program for computing leaf conductance and detecting curve's change


## Install Python version if needed

[Anaconda](https://www.anaconda.com/products/individual)
[Miniconda](https://docs.conda.io/en/latest/miniconda.html)


## Download full folder from git

1. Direct download

From the green box  named 'clone' in the right corner > download .zip

2. From the terminal

>
> git clone https://github.com/xbouteiller/DetectEvent.git
>



## Install dependencies

>
> pip install -r requirements.txt 
>


## Install package

Open a terminal in the DetectEvent folder, then :

>
> python setup.py develop
>


## Program Execution

Copy the file **Leaf_exec.py** in the desired folder

Then open a terminal 


>
> python Leaf_exec.py
>

## Program possibilities

### RMSE approach

- Objective is to detect curve's changing point (i.e the point where the curve shift from an exponential function to a linear function)
- Exponential function is fitted from left to right and linear function is fitted from right to left
- Root Mean Squared Error (RMSE) is computed and it is excpected that errors cross  approximately at he curve's changing point
- Several computed parameters are extracted and saved within a csv file (as Gmin)

### Robust differential method

- Use a robust differential method (Chartrand, 2011) to compute conductance from raw data 
- Conductance is further differentiated in order to detect peak


## Data

Data must be stored within files
For a better files recognition, first row of the csv file should contain the string conductance otherwise all csv from a folder will be parsed

Columns name should be as below

### Quantitative columns 
- weight_g : leaf weight as a function of time (g)
- T_C : temperature (°C)
- RH : 
- Patm : atmospheric pressure (KPa)
- Area_m2 : area of the leaf (m2)

### Qualitative columns 
- campaign : campaign name
- sample_ID : name of the sample (several samples can be in the same csv file)

### Date

- date_time : time of the experiment (best with the format YEAR/MONTH/DAY HOUR:MINUTE )


## References

> Chartrand, Rick. (2011). Numerical Differentiation of Noisy, Nonsmooth Data. ISRN Appl. Math.. 2011. 10.5402/2011/164564. 
[Original paper](https://www.hindawi.com/journals/isrn/2011/164564/)
[GitHub](https://github.com/xbouteiller/tvregdiff)