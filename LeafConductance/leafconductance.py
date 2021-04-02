import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from loess.loess_1d import loess_1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from LeafConductance.tvregdiff import log_iteration, TVRegDiff 
import pandas as pd
from scipy.optimize import curve_fit
import time
import sys
from scipy.signal import find_peaks
from scipy import stats
import os
from matplotlib import colors as mcolors


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

print('------------------------------------------------------------------------')
print('---------------                                    ---------------------')
print('---------------            LeafConductance         ---------------------')
print('---------------                  V1.5              ---------------------')
print('---------------                                    ---------------------')
print('------------------------------------------------------------------------')
time.sleep(0.5)

# num_col = ['weight_g','T_C','RH', 'Patm', 'Area_m2']
# group_col=['campaign', 'sample_ID', 'Treatment', 'Operator']
# date=['date_time']

# error_proof
# GLOBAL VARIABLES

SEP = ','

#TIME_COL = 'date_time'
#SAMPLE_ID = 'sample_ID'
#YVAR = 'weight_g'
# T = 'T_C'
# RH = 'RH'
# PATM = 'Patm'
# AREA = 'Area_m2'

WIND_DIV = 8
LAG_DIV = WIND_DIV * 200
BOUND = 'NotSet'
FRAC_P = 0.1
PAUSE_GRAPH = 8
DELTA_MULTI = 0.01

# ITERN=10000
# ALPH=1e9#1000
# EP=1e-6
#KERNEL='sq'#'sq'#abs'#'abs'
THRES = 50 #3

class ParseFile():
    import pandas as pd
    import numpy as np

    

    def __init__(self, path, skipr=1, sepa= SEP, encod = "utf-8"):
        '''
        initialization
        path of the file
        skipfoot : number of rows to skip at the end of the txt file

        portability : allow manual definition of skiprows and delimiter
                      test the file format and provide the good function for reading the file
        '''

        import pandas as pd
        import numpy as np
        try:
            self.file = pd.read_csv(path, skiprows=skipr)
        except:
            self.file = pd.read_csv(path, skiprows=skipr, sep=sepa, encoding=encod)
    

    def clean_file(self):
        '''
        clean the file

        Currently do nothing but can be adapted in order to clean file individually
        '''
        import re
        import pandas as pd
        import numpy as np


        return self.file



class ParseTreeFolder(): 


    def _get_valid_input(self, input_string, valid_options):
        '''
        useful function in order to ask input value and assess that the answer is allowed

        input_string : question
        valid_options : authorized answers
        '''
        input_string += "({}) ".format(", ".join(valid_options))
        response = input(input_string)
        while response.lower() not in valid_options:
            response = input(input_string)
        return response


    def __init__(self,                
                time_col, 
                sample_id,
                yvar,
                temp,
                rh,
                patm,
                area,
                iter_n,
                alpha,
                epsilon,
                diff_method,
                transfo_rmse,
                transfo_diff,
                fit_exp_rmse,
                fit_exp_diff,
                rwc_sup,
                rwc_inf):

        import time
        # super().__init__()
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename, askdirectory

        
        self.TIME_COL = time_col
        self.SAMPLE_ID = sample_id
        self.YVAR = yvar
        self.T = temp
        self.RH = rh
        self.PATM = patm
        self.AREA = area

        self.ITERN = iter_n
        self.ALPH = alpha
        self.EP = epsilon
        self.KERNEL = diff_method

        self.transfo_rmse = transfo_rmse
        self.transfo_diff = transfo_diff

        self.fit_exp_rmse = fit_exp_rmse
        self.fit_exp_diff = fit_exp_diff

        self.rwc_sup = rwc_sup
        self.rwc_inf = rwc_inf

        # global class variables
        self.global_score = []
        self.Conductance = False
        self.remove_outlier = False

        print('''
        WELCOME TO LEAFCONDUCTANCE

        What do you want to do ?

        1: Parse files from a folder
        2: Select some ID from a file
        ''')
        self.file_or_folder = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2'))

        # fixed value for self.file_or_folder attribute, to be cleaned in the future
       
        # self.file_or_folder = self._get_valid_input('Type 1 to start : ', ('1'))
        if self.file_or_folder== '1':
            ################################################### REACTIVATE
            root_path = os.getcwd()
            Tk().withdraw()
            folder = askdirectory(title = 'What is the root folder that you want to parse ?',
                                  initialdir = root_path)
            # folder = '/home/xavier/Documents/development/DetectEvent/data'
            #####################################################""
            self.path = folder
            print('\n\n\nroot path is {}'.format(self.path))
            ################################################### REACTIVATE
            print('''
            which method do you want to use for detecting conductance files ?

            1: Detect all CSV files 
            2: Detect string 'CONDUCTANCE' in the first row
            ''')
            self.method_choice = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2'))
            # self.method_choice = '2' 
            ################################################### REACTIVATE
            print('\n\n\nfile 1 path is {}'.format(self.path))

        if self.file_or_folder== '2':
            Tk().withdraw()
            file = askopenfilename(title='What is the file that you want to check ?')
            self.path = file.replace('/','/')
            print('\n\n\nfile path is {}'.format(self.path))   

            print('''
            which method do you want to use for detecting conductance files ?

            1: Detect all CSV files 
            2: Detect string 'CONDUCTANCE' in the first row
            ''')
            self.method_choice = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2'))        


        # options allowed for the action method
        # in the future it could be useful to add a combo method that would do '1' & '2' in the same time
        self.choices = {
        "1": self.change_detection,
        "2": self.robust_differential,
        "3": self._quit,
        # "4": self.erase,
        # "5": self.extract_strings_and_nums
        }


    def _listdir_fullpath(self, p, s):
        '''
        method for creating a list of csv file
        '''
        import os
        import re
        d=os.path.join(p, s)
        return [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.csv')]

    def _detect_cavisoft(self, p, s):
        '''
        method for creating a list of conductance files, based on the detection of the string "conductance"
        '''
        import pandas as pd
        import re
        import os
        d=os.path.join(p, s)

        return [os.path.join(d, f) for f in os.listdir(d) if\
                f.endswith('.csv') and (re.search(r'conductance', pd.read_csv(os.path.join(self.path, f),sep=SEP,nrows=0).columns[0].lower()) )]

    
    def parse_folder(self):
        '''
        parse the folder tree and store the full path to target file in a list
        '''
        import os
        import time
        import re
        import pandas as pd

        if self.file_or_folder=='2':
            self.listOfFiles=[[self.path]]


        if self.file_or_folder=='1':
            file_root=[]
            self.listOfFiles = []

            # method with csv detection
            if self.method_choice == '1':
                try:
                    # basedir
                    file_root = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith('.csv')]
                    self.listOfFiles.append(file_root)
                    #print(file_root)
                except:
                    print('no file detected within root directory')
                    pass

                try:
                    #subfolders
                    for pa, subdirs, files in os.walk(self.path):
                        for s in subdirs:
                            self.listOfFiles.append(self._listdir_fullpath(p=pa, s=s))
                except:
                    print('no file detected within childs directory')
                    pass
            
            # method with conductance detection
            if self.method_choice == '2':
                try:
                    # base dir
                    file_root = [os.path.join(self.path, f) for f in os.listdir(self.path) if\
                    f.endswith('.csv') and (re.search(r'conductance', pd.read_csv(os.path.join(self.path, f),sep=SEP,nrows=0).columns[0].lower()) )]
                    self.listOfFiles.append(file_root)
                    #print(file_root)
                except:
                    print('no file detected within root directory')
                    pass

                try:
                    #subfolders
                    for pa, subdirs, files in os.walk(self.path):
                        for s in subdirs:
                            self.listOfFiles.append(self._detect_cavisoft(p=pa, s=s))
                except:
                    print('no file detected within childs directory')
                    pass


            print('\n')
            try:
                [print("- find : {0} matching files in folder {1}".format(len(i),j)) for i,j in zip(self.listOfFiles, range(1,len(self.listOfFiles)+1))]
            except:
                print('no files detected at all')
                pass

            time.sleep(1)

            return self.listOfFiles

    def print_listofiles(self):
        '''
        print full path to target file
        '''
        # Print the files
        for elem in self.listOfFiles:
            print(elem)


    def display_menu(self):
        print("""
        --------------------
        -----   MENU   -----
        --------------------

        List of actions

        1. Detect changes in curve (RMSE approach)
        2. Compute conductance (robust differential approach)
        3. Exit
        """)

    def _quit(self):
        print("Thank you for using your LeafConductance today.\n")
        sys.exit(0)

    def run(self):
        '''Display the menu and respond to choices.'''
        
        while True:
            self.display_menu()
            choice = input("Enter an option: ")

            # redirection to the self.choices attribute in the __init__
            action = self.choices.get(choice)

            if action:
                action()
            else:
                print("{0} is not a valid choice".format(choice))
                self.run()


    def _min_max(self, X): 
        '''
        min max scaler, return array of scaled values comprised between 0 & 1
        '''
        X_scaled = (X-np.min(X)) / (np.max(X) -  np.min(X))
        return X_scaled

    def _RMSE(self, Ypred, Yreal):
        '''
        compute the root mean squared error between two arrays of the same length
        '''
        rmse = np.sqrt(np.sum(np.square(Ypred-Yreal))/np.shape(Ypred)[0])
        return rmse

    def _fit_and_pred(self,X, y, mode, mode2='raw', *args):
        '''
        fit a linear regression or a exponential regression between 2 arrays X & Y
        function for the exponential regression should be defined in the self._func() method
        return the RMSE of the fit

        mode : linear or exp
        mode2 : raw signal or derivated signal


        '''



        if mode == 'linear':
            Xarr = np.array(X).reshape(-1,1)
            yarr = np.array(y).reshape(-1,1)
            
            # print('log y')
            #print('log y')
            if mode2 == 'raw':
                # log data for the linear phase of the fit only if 2 or 4 were chosen
                if self.transfo_rmse == '2' or self.transfo_rmse == '4':
                    # Xarr = (Xarr+0.1)
                    yarr = np.log(yarr+1)# 
                    #print('transfo y lin')
                    # 
                    # 
            # for differentiated data

            if mode2 == 'diff':
                # log data for the linear phase of the fit only if 2 or 4 were chosen
                if self.transfo_diff == '2' or self.transfo_diff == '4':
                    yarr =  1/np.exp(yarr) #np.sqrt(yarr)# 
                    #print('transfo y lin')
            
            if mode2 == 'raw' or mode2 == 'diff':              
                reg = LinearRegression().fit(Xarr, yarr)
                pred = reg.predict(Xarr)

        if mode == 'exp':
            # Xarr = np.array(X).reshape(-1) # REMOVE THE , -1 WARNING
            # yarr = np.array(y).reshape(-1)

            # print('sqrt y')
            # yarr = np.sqrt(yarr)

            #--------------------------------------------------------------------
            #Xarr = np.array(X)#.reshape(-1,1) # 
            #yarr = np.array(y)#.reshape(-1,1)
            #sample_weight = np.ones(yarr.size)
            
            # # print(yarr)
            # if mode2 == 'raw':
            #     if self.transfo_rmse == '3' or self.transfo_rmse == '4':
            #         yarr = 1/np.exp(yarr)
            # if mode2 == 'diff':
            #     if self.transfo_diff == '3' or self.transfo_diff == '4':
            #         yarr = 1/np.exp(yarr)

            # print(yarr)

            if mode2 == 'raw':

                # linear regression OR exp fit A*exp-B*t can be used
                if self.fit_exp_rmse == '2':
                    #print('linear reg')
                    Xarr = np.array(X).reshape(-1,1) # 
                    yarr = np.array(y).reshape(-1,1)
                    
                    # 1/exp data for the exp phase of the fit only if 3 or 4 were chosen
                    if self.transfo_rmse == '3' or self.transfo_rmse == '4':
                        #print('transfo y exp')
                        # Xarr = 1/(Xarr+0.1) ## TRY TO INVERT X 
                        yarr = 1/np.exp(yarr)#1/np.exp(yarr)              

                    reg = LinearRegression().fit(Xarr, yarr)
                    pred = reg.predict(Xarr)

                # linear regression OR exp fit A*exp-B*t can be used
                elif self.fit_exp_rmse == '1':
                    #print('CF reg')

                    Xarr = np.array(X).reshape(-1) # 
                    yarr = np.array(y).reshape(-1)
                       
                    # 1/exp data for the exp phase of the fit only if 3 or 4 were chosen
                    if self.transfo_rmse == '3' or self.transfo_rmse == '4':
                        #print('transfo y exp')
                        yarr = 1/np.exp(yarr)#1/np.exp(yarr) 

                    reg = curve_fit(self._func, Xarr, yarr, bounds=args[0])[0]
                    A, B = reg     
                    pred = A * np.exp(-B * Xarr)

            # for differentiated data
            if mode2 == 'diff':
                
                # linear regression OR exp fit A*exp-B*t can be used
                if self.fit_exp_diff == '2':
                    #print('linear reg')

                    Xarr = np.array(X).reshape(-1,1) # 
                    yarr = np.array(y).reshape(-1,1)
                    if self.transfo_diff == '3' or self.transfo_diff == '4':
                        yarr = 1/np.exp(yarr)
                        #print('transfo y exp')
                    
                    reg = LinearRegression().fit(Xarr, yarr)
                    pred = reg.predict(Xarr)
                
                # linear regression OR exp fit A*exp-B*t can be used
                elif self.fit_exp_diff == '1':
                    #print('CF reg')

                    Xarr = np.array(X).reshape(-1) # 
                    yarr = np.array(y).reshape(-1)

                    if self.transfo_diff == '3' or self.transfo_diff == '4':
                        yarr = 1/np.exp(yarr)
                        #print('transfo y exp')

                    reg = curve_fit(self._func, Xarr, yarr, bounds=args[0])[0]
                    A, B = reg     
                    pred = A * np.exp(-B * Xarr) 

        rmse = self._RMSE(pred, yarr)
        return rmse
    
    def _sliding_window_pred(self, X, y, window, lag, mode, b=BOUND, mode2='raw'):
        '''
        Define a sliding window where the _fit_and_pred() method is applied
        from start --> end : exp fit
        from end --> linear fit

        mode : linear or exp
        mode2 : raw signal or derivated signal
        '''


        self._print_fit_parameters( mode2 = mode2)

        # parameters to avoid exceed data boundaries
        Xend = np.shape(X)[0]
        Xmax = np.shape(X)[0]-lag-1



        # coordinates of the sliding window

        if mode2=='raw':
            start = np.arange(window, Xend-window, lag)
        if mode2=='diff':
            start = np.arange(window, Xend-window, lag)
        print(mode, mode2)
        

        if mode == 'linear':
            if mode2 == 'raw':
            # X position of sliding window increment
                mean_start = X[[int(Xend-s) for s in start]]
                score = [self._fit_and_pred(X[Xend-s:Xend], y[Xend-s:Xend], mode) 
                        for s in start]    #[::-1]

            if mode2 == 'diff':
                # mean_start = X[[int(Xend-start[i]) for i,_ in enumerate(start) if i < len(start)-1]]
                mean_start = X[[int(Xend-s) for s in start]]

                # print('length of start vector:', len(start))
                # for i, _ in enumerate(start):
                #     if i < len(start)-1:
                #         print('i:', i)
                #         print('lin regression between {}-{}'.format(Xend-start[i+1],Xend-start[i]))
                
                score = [self._fit_and_pred(X[Xend-s:Xend], y[Xend-s:Xend], mode, mode2) 
                        for s in start]    

                # print('length of lin score: ', len(score))
                # print('length of lin mean_start: ', len(mean_start))
        if mode == 'exp':
            # X position of sliding window increment
            if mode2=='raw':
                mean_start = X[start]
            if mode2=='diff':
                # mean_start = X[[start[i] for i,_ in enumerate(start) if i < len(start)-1]]
                mean_start = X[start]
            # if no parameters are provided for constraining the timit of A & B parameters for the exp fitting
            if BOUND == 'NotSet':
                try:
                    # find constrained parameters for the exp fitting
                    if mode2 == 'raw':
                        reg = self._detect_b( X[lag:lag+(window*2)], y[lag:lag+(window*2)], mode)
                    if mode2 == 'diff':
                        print('size of y ',y.size)
                        tinf = lag/y.size#1/771
                        tsup = lag+(window*2)/y.size #0.3#(1/771)*193
                        print('default window value {}-{}'.format(lag, lag+(window*2)))
                        print('window for estimating bound {}-{}\n'.format(int(y.size * tinf),int(y.size * tsup)))
                        reg = self._detect_b( X[int(y.size * tinf): int(y.size * tsup)], y[int(y.size * tinf): int(y.size * tsup)], mode, mode2=mode2)
                   
                    Aa, Bb = reg
                    print('Estimated bound # are : {}-{}'.format(Aa,Bb))

                    if mode2 == 'raw':                      
                        bound = ([Aa-0.015*Aa,Bb/1.05],[Aa+0.015*Aa, Bb*1.05])
                    if mode2 == 'diff':
                        ssub = 0.4
                        sdiv = 1.4                       
                        print('mode is {}\nrestricted bound values'.format(mode2))
                        bound =  ([0,1/1000000],[100, 1/100])#([Aa-ssub*Aa,Bb/sdiv],[Aa+ssub*Aa, Bb*sdiv])
                except:
                    # if the detection failed, provide wide parameters values
                    bound = ([0,1/1000000],[100, 1/100])
            else:
                bound=BOUND
            print('bound : ', bound)

            # score = [self._fit_and_pred(X[0:s], y[0:s], mode, mode2, 'lin', bound) 
            #                 for s in start]
            # print('score', score)
            # input()
            # [self._fit_and_pred(X[0:s], y[0:s], mode, mode2, bound) 
            #                 for s in start]
            try:
                # do the exp fit
                # curve fit from scipy is embedded within _fit_and_pred
                
                # it is the same curve fit method both for diff signal and raw signal
                if mode2=='raw' or mode2 == 'diff':
                    score = [self._fit_and_pred(X[0:s], y[0:s], mode, mode2, bound) 
                            for s in start]

                # print('length of exp score: ', len(score))
                # print('length of exp mean_start: ', len(mean_start))
            except:
                #raise Exception('Failed to fit Exponential curve')

                print('\n+++++++++++++++++++++++++++++++++++')
                print('  Failed to fit Exponential curve  ')
                print('     You should discard values  ')
                print('+++++++++++++++++++++++++++++++++++\n')
                time.sleep(1)
                # if failed, return array of 1
                score = np.repeat(1, len(start))
        score = self._min_max(score)
        #print('{} score'.format(mode), score)
        return score, mean_start

    def _print_fit_parameters(self, mode2):

        if mode2 == 'raw':
            print('\n-------------------\nRaw data')
            if self.transfo_rmse == '1':
                print('no transformation applied en data')
            elif self.transfo_rmse == '2':
                print('lin part was log transformed')
            elif self.transfo_rmse == '3':
                print('exp part was 1/exp transformed')
            else:
                print('lin part was log transformed & exp part was 1/exp transformed')

            if self.fit_exp_rmse == '1':
                print('\nA+exp-B*t was fitted on exponential part\n')
            else :
                print('linear model was fitted on exponential part\n')
       
        if mode2 == 'diff':
            print('\n-------------------\nDifferentiated data')
            if self.transfo_diff == '1':
                print('no transformation applied en data')
            elif self.transfo_diff == '2':
                print('lin part was log transformed')
            elif self.transfo_diff == '3':
                print('exp part was 1/exp transformed')
            else:
                print('lin part was log transformed & exp part was 1/exp transformed')

            if self.fit_exp_diff == '1':
                print('\nA+exp-B*t was fitted on exponential part\n')
            else :
                print('linear model was fitted on exponential part\n')
        
        time.sleep(1)
            
            


    def _func(self, x, a, b):
        # equation for the exp fitting
        return a * np.exp(-b * x) 
        #return (-1/(a * np.exp(-b * x))) + 1

    def _func_d(self, x, a, b):
        # equation for the exp fitting
        return a*np.exp(-b * x) 


    def _func_lin(self, x, a, b):
        return a * x + b 
    
    def _detect_b(self, X, y, mode, mode2='raw'):
        '''
        function used for detecting boundaries of parameters for the exp fitting
        '''
        Xarr = np.array(X).reshape(-1)
        yarr = np.array(y).reshape(-1)
        if mode == 'exp':
            if mode2 == 'raw':
                reg = curve_fit(self._func, Xarr, yarr)[0]
            if mode2 == 'diff':
                reg = curve_fit(self._func_d, Xarr, yarr)[0]
        return reg

    def _dcross(self, Yl, Ye):
        '''
        detect he index where two signals are crossing each others. Based on the detection in a shift in the sign of the difference
        '''
        idx = np.argwhere(np.diff(np.sign(Yl - Ye))).flatten() 
        return idx

    def _compute_slope(self, Xidx1, interval = False, *args, **kwargs):
        '''
        fit a linear regression on a slice of a signal and return slope, intercept, rsqquared and fitted values
        '''         
        # this is useful for slicing X value returned by ginput
        if len(Xidx1)>1:
            Xidx1 = Xidx1[0]  

        print('Xidx1: ', Xidx1)     

        if interval:                       
            Xidx2 = args[0]
            if len(Xidx2)>1:
                Xidx2 = Xidx2[0]
            print('Xidx2: ', Xidx2)   
            X = self.Xselected[(self.Xselected >= Xidx1) & (self.Xselected <= Xidx2)].copy()
            y = self.yselected[(self.Xselected >= Xidx1) & (self.Xselected <= Xidx2)].copy()
        else:
            X = self.Xselected[self.Xselected >= Xidx1].copy()
            y = self.yselected[self.Xselected >= Xidx1].copy()

        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

        fitted_values = slope*X + intercept
        rsquared = r_value**2

        return slope, intercept, rsquared, fitted_values, X

    def _compute_gmin_mean(self, df, slope, t1, t2 = None):
        '''
        compute gmin on a slice of a signal
        use the slope value
        '''
        if t2 is None:
            df = df[df['delta_time']>= t1].copy()
        if t2 is not None:            
            df = df[(df['delta_time']>= t1) & (df['delta_time']<= t2)].copy()

        k= (slope/18.01528)*(1000/60) #ici c'est en minutes (60*60*24)

        #Calcul VPD en kpa (Patm = 101.325 kPa)
        VPD =0.1*((6.13753*np.exp((17.966*np.mean(df[self.T].values)/(np.mean(df[self.T].values)+247.15)))) - (np.mean(df[self.RH].values)/100*(6.13753*np.exp((17.966*np.mean(df[self.T].values)/(np.mean(df[self.T].values)+247.15)))))) 

        #calcul gmin mmol.s
        gmin_ = -k * np.mean(df[self.PATM].values)/VPD

        #calcul gmin en mmol.m-2.s-1
        gmin = gmin_ / np.mean(df[self.AREA].values)

        print('gmin_mean: ', gmin)

        return gmin, [k, VPD, np.mean(df[self.T].values), np.mean(df[self.RH].values), np.mean(df[self.PATM].values), np.mean(df[self.AREA].values)]



    def _detect_crossing_int(self, Yexp, Ylin, Xl, Xe, df, mode = 'raw'):
        '''
        detection of the crossing point of the rmse error from the exp & linear fitting
        yexp & ylin : rmse from the sliding window
        '''
        gmin_mean=''
        list_of_param=['', '', '', '', '', '']

        Ylin=np.array(Ylin)
        Yexp=np.array(Yexp)
        Xl=np.array(Xl)  
        Xe=np.array(Xe) 
        # detect the index of crossing: needs to reverse the ylin
        idx = self._dcross(Ylin[::-1], Yexp)
        Xidx=Xe[idx]    

        #print('Xidx : ', Xidx)
        idx_int = [[i, i+1] for i in idx]
        Xidx_int = [[Xe[i], Xe[i+1] ]for i in idx]

        if mode == 'diff':
            self.yselected = df['gmin']
        
        if len(Xidx)==1:
            # if only one crossing is detected
            if mode == 'raw':
                Yidx=self.Ysmooth[self.Xselected == Xidx]
            if mode == 'diff':
                Yidx=self.yselected[self.Xselected == Xidx]
        else:
            # if more than 1 crossing is detected
            a = []
            if mode == 'raw':
                [a.append(self.Ysmooth[self.Xselected == i]) for i in Xidx]
                Yidx = a
            if mode == 'diff':
                [a.append(self.yselected[self.Xselected == i]) for i in Xidx]
                Yidx = a

        if mode == 'raw' or mode != 'raw':
            # plot the figure of the detected crossing sections        
            fig, ax1 = plt.subplots()
            plt.title(self.sample)

            color = 'tab:blue'
            # ax1 : raw signal
            ax1.set_xlabel('time (min)')
            ax1.set_ylabel(self.sample, color=color)
            if mode == 'raw':
                ax1.plot(self.Xselected, self.yselected, color=color, linestyle='-', marker='.', label = 'Weight (g)')
            if mode == 'diff':
                ax1.plot(self.Xselected, self.yselected, color=color, linestyle='-', marker='.', label = 'Differentiated signal (G)')
            ax1.tick_params(axis='y', labelcolor=color)
            color = 'tab:red'
            # ax1 : smoothed signal
            if mode != 'diff':
                ax1.plot(self.Xselected, self.Ysmooth, color=color, lw=2, linestyle='-', label = 'smooth')        
            
            # crossing points : printed as red dot
            ax1.plot(Xidx, Yidx, 'ro', markersize=8)

            ax1.hlines(xmin=0,xmax=self.Xselected[-1],y=Yidx, color='red', lw=0.8, linestyle='--')
            ax1.vlines(ymin=np.min(self.yselected),ymax = np.max(self.yselected),x=Xidx, color='red', lw=0.8, linestyle='--')
            ax1.legend(loc='upper right')

            # compute the slope if only 1 crossing point is detected
            if len(Xidx)==1:
                slope, intercept, rsquared, fitted_values, Xreg = self._compute_slope(Xidx1=Xidx)
                ax1.plot(Xreg, fitted_values, c = colors['black'], lw = 2)
                gmin_mean, list_of_param = self._compute_gmin_mean(df=df, slope=slope, t1=Xidx[0], t2 = None)
            else:
                print('more than 1 crossing point were detected')

            # ax2: plot he two RMSE 
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:green'
            ax2.set_ylabel('RMSE')  # we already handled the x-label with ax1
            ax2.plot(Xl, Ylin, color=color,  marker='.', label = 'RMSE lin')
            #ax2.tick_params(axis='y', labelcolor=color)
            color = 'tab:orange'
            ax2.set_ylabel('RMSE', color=color)  # we already handled the x-label with ax1
            ax2.plot(Xe, Yexp, color=color, marker='.', label = 'RMSE exp')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.legend(loc='right')
            fig.tight_layout()

            try:
                self.fig_folder
            except:
                self.fig_folder = 'None'

            if self.fig_folder == 'None':
                starting_name = 'output_fig'
                i = 0
                while True:
                    i+=1
                    fig_folder = starting_name+'_'+str(i)
                    if not os.path.exists(fig_folder):
                        os.makedirs(fig_folder)
                        os.makedirs(fig_folder+'/'+'rmse')
                        os.makedirs(fig_folder+'/'+'diff')
                        os.makedirs(fig_folder+'/'+'rwc')
                        break
                self.fig_folder = fig_folder                

            if mode == 'raw':
                figname = self.fig_folder + '/' + 'rmse' + '/' + self.sample + '.png'
                plt.savefig(figname, dpi = 420, bbox_inches = 'tight')
            if mode == 'diff':
                figname = self.fig_folder + '/' + 'diff' + '/' + self.sample + '.png'
                plt.savefig(figname, dpi = 420, bbox_inches = 'tight')

            # close the graph on a click
            # plt.pause(PAUSE_GRAPH)
            plt.waitforbuttonpress(0)

            # input()
            plt.close()   

        

            print('\nInterval method')
            for i in np.arange(0,len(idx)):
                print('detected changes between times : {} - {}'.format(Xidx_int[i][0], Xidx_int[i][1]))
            
            if mode == 'raw':
                # When more than one crossing point are detected
                # future change the index of the questions
                if len(Xidx)>1:           
                    print('''
                            More than 1 crossing point were detected, you have to chose between :
                            
                            2: No, discard
                            3. Select values manually on graph
                            ''')                
                    what_to_do = self._get_valid_input('What do you want to do ? Choose one of : ', ('2','3'))
                        
                else :
                    print('''
                        Do you want to keep the crossing value ?

                        1: Yes, save
                        2: No, discard
                        3. Select values manually on graph
                        ''')
                    what_to_do = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2','3'))

                if what_to_do=='2':
                    self.global_score.append([self.sample,'Discarded','Discarded','Discarded','Discarded', ['', '', '', '', '', '']])
                if what_to_do=='1':
                    self.global_score.append([self.sample, Xidx, slope, rsquared, gmin_mean, list_of_param])
                if what_to_do=='3':
                    while True:
                        try:
                            # future allow only selection of 1 or 2 points                        
                            _Npoints = int(self._get_valid_input('How many points do you want to select ? ',('1', '2')) or 1)              
                            break
                        except ValueError:
                            print("Oops!  That was no valid number.  Try again...")

                    First_pass = 0
                    while First_pass < 2:
                        # first pass selection of the points
                        # second pass fit the linear regression & compute the slope
                        fig, ax1 = plt.subplots()
                        plt.title(self.sample)
                        color = 'tab:blue'
                        ax1.set_xlabel('time (min)')
                        ax1.set_ylabel(self.sample, color=color)
                        ax1.plot(self.Xselected, self.yselected, color=color, linestyle='-', marker='.', label = 'Weight (g)')
                        ax1.tick_params(axis='y', labelcolor=color)
                        color = 'tab:red'
                        if mode != 'diff':
                            ax1.plot(self.Xselected, self.Ysmooth, color=color, lw=2, linestyle='-', label = 'smooth')        
                        ax1.plot(Xidx, Yidx, 'ro', markersize=8)
                        ax1.hlines(xmin=0,xmax=self.Xselected[-1],y=Yidx, color='red', lw=0.8, linestyle='--')
                        ax1.vlines(ymin=np.min(self.yselected),ymax = np.max(self.yselected),x=Xidx, color='red', lw=0.8, linestyle='--')
                        ax1.legend(loc='upper right')
                                

                        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                        color = 'tab:green'
                        ax2.set_ylabel('RMSE')  # we already handled the x-label with ax1
                        ax2.plot(Xl, Ylin, color=color,  marker='.', label = 'RMSE lin')
                        #ax2.tick_params(axis='y', labelcolor=color)
                        color = 'tab:orange'
                        ax2.set_ylabel('RMSE', color=color)  # we already handled the x-label with ax1
                        ax2.plot(Xe, Yexp, color=color, marker='.', label = 'RMSE exp')
                        ax2.tick_params(axis='y', labelcolor=color)
                        ax2.legend(loc='right')
                        fig.tight_layout()

                        if First_pass == 0:
                            selected_points = fig.ginput(_Npoints)
                            if _Npoints==1:
                                slope, intercept, rsquared, fitted_values, Xreg = self._compute_slope(Xidx1=selected_points[0])
                                gmin_mean, list_of_param = self._compute_gmin_mean(df=df, slope=slope, t1=selected_points[0][0], t2 = None)                       
                            elif _Npoints==2:
                                slope, intercept, rsquared, fitted_values, Xreg = self._compute_slope(selected_points[0], True, selected_points[1]) 
                                gmin_mean, list_of_param = self._compute_gmin_mean(df=df, slope=slope, t1=selected_points[0][0], t2 = selected_points[1][0])                         
                            else:
                                print('unable to fit regression')

                        else:                    
                            if _Npoints==1:                       
                                ax1.plot(Xreg, fitted_values, c = colors['black'], lw = 2)
                            elif _Npoints==2:                        
                                ax1.plot(Xreg, fitted_values, c = colors['black'], lw = 2)
                            else:
                                print('unable to fit regression')

                        try:
                            self.fig_folder
                        except:
                            self.fig_folder = 'None'

                        if self.fig_folder == 'None':
                            starting_name = 'output_fig'
                            i = 0
                            while True:
                                i+=1
                                fig_folder = starting_name+'_'+str(i)
                                if not os.path.exists(fig_folder):
                                    os.makedirs(fig_folder)
                                    os.makedirs(fig_folder+'/'+'rmse')
                                    os.makedirs(fig_folder+'/'+'diff')
                                    os.makedirs(fig_folder+'/'+'rwc')
                                    break
                            self.fig_folder = fig_folder                

                        if mode == 'raw':
                            figname = self.fig_folder + '/' + 'rmse' + '/' + self.sample + '.png'
                            plt.savefig(figname, dpi = 420, bbox_inches = 'tight')
                        if mode == 'diff':
                            figname = self.fig_folder + '/' + 'diff' + '/' + self.sample + '.png'
                            plt.savefig(figname, dpi = 420, bbox_inches = 'tight')

                        plt.waitforbuttonpress(0)      
                        plt.close()
                        First_pass+=1

                
                    print('\nSelected points at time : ', ' '.join([str(i[0]) for i in selected_points ]))
                    print('\n')
                    self.global_score.append([self.sample, [i[0] for i in selected_points ], slope, rsquared, gmin_mean, list_of_param])  


        print('gs',self.global_score)

        return idx, Xidx, Xidx_int 
    
    def _change_det(self, df, COL_Y='standard', mode = 'raw'):
        '''
        detect the crossing point
        '''
        
        # initialization of the sliding window
        if mode == 'raw':
            if df.shape[0] < 100:
                _wind = int(df.shape[0]/6)
                _lag = 1# int(df.shape[0]/4)
            else:
                _wind = max(int(df.shape[0]/WIND_DIV),int(1))
                _lag = max(int(df.shape[0]/LAG_DIV),int(1))

        # useful for detecting peak within derivated signal
        if mode =='diff':
            if df.shape[0] < 100:
                _wind = int(df.shape[0]/10)
                _lag = 1# int(df.shape[0]/4)
            else:
                _wind = max(int(df.shape[0]/WIND_DIV),int(1))
                _lag = max(int(df.shape[0]/LAG_DIV),int(1))

        _X = df['delta_time'].copy().values

        # eventually col_y could be modified e.g. with argparser (YVAR = 'weight_g' by defeult)
        # here the tvregdiff return gmin instead of 'standard'
        if COL_Y == 'standard':
            _y = df[self.YVAR].copy().values
        else:
            print("using column: {} as y".format(COL_Y))
            _y = df[COL_Y].copy().values

        #print(df.head())
        #input()


        # create a df_vlaue attritute
        
        if (self.df_value is None) or (mode == 'diff'):
            self.df_value = pd.DataFrame(columns = df.columns)
        
        # append to df value
        self.df_value = pd.concat([self.df_value, df], axis = 0, ignore_index = True)

        # sliding window & compute rmse for each window
        score_l, mean_start_l = self._sliding_window_pred(_X, _y, window=_wind, lag=_lag, mode = 'linear', mode2 = mode)
        score_e, mean_start_e = self._sliding_window_pred(_X, _y, window=_wind, lag=_lag, mode = 'exp', mode2 = mode)

        # print("Before exception ", score_l, mean_start_l, score_e, mean_start_e )

        # create an empty df for rmse values
        if self.df_rmse is None:
            self.df_rmse = pd.DataFrame(columns = ['Sample', 'RMSE_lin', 'Time_lin', 'RMSE_exp', 'Time_exp'])        
        
        # concat to rmse df the current file rmse
        df_temp_rmse = pd.DataFrame({'Sample':self.sample, 'RMSE_lin':score_l, 'Time_lin':mean_start_l, 'RMSE_exp':score_e, 'Time_exp':mean_start_e})        
        self.df_rmse = pd.concat([self.df_rmse, df_temp_rmse], axis = 0, ignore_index = True)
        
        # plt.scatter(mean_start_e,score_e)
        # plt.scatter(mean_start_l,score_l)
        # plt.show()

        # print(df.dtypes)
        # self._detect_crossing_int(Ylin=score_l, Yexp=score_e, Xl= mean_start_l, Xe= mean_start_e, df = df, mode = mode)

        try:
            idx, Xidx, Xidx_int = self._detect_crossing_int(Ylin=score_l, Yexp=score_e, Xl= mean_start_l, Xe= mean_start_e, df = df, mode = mode) #Yexp, Ylin, Xl, Xe
            return idx, Xidx, Xidx_int
        except OSError as err:
            print("OS error: {0}".format(err))
            try:
                plt.close()
            except:
                pass
            self.global_score.append([self.sample, 'Failed', 'Failed', 'Failed', 'Failed', ['', '', '', '', '', '']])

        except ValueError as verr:
            print("ValueError error: {0}".format(verr))
            try:
                plt.close()
            except:
                pass
            self.global_score.append([self.sample, 'Failed', 'Failed', 'Failed', 'Failed', ['', '', '', '', '', '']])

        except TypeError as typ:
            print("TypeError error: {0}".format(typ))
            try:
                plt.close()
            except:
                pass
            self.global_score.append([self.sample, 'Failed', 'Failed', 'Failed', 'Failed', ['', '', '', '', '', '']])

        except:
            print("Unexpected error:", sys.exc_info()[0])
            print('Detect crossing failed, return Failed value within the data frame')
            try:
                plt.close()
            except:
                pass
            self.global_score.append([self.sample, 'Failed', 'Failed', 'Failed', 'Failed', ['', '', '', '', '', '']])


    def _smoother(self, ex, end, fr, delta_multi):
        delt = delta_multi * ex.shape[0]
        Ysmooth = lowess(exog = ex, endog = end, frac = fr, delta = delt, return_sorted = False)
        return Ysmooth

    def _turn_on_off_remove_outlier(self, state):
        self.remove_outlier=state

    def _detect_outlier(self, df, thres):
        ''' function for dtetecting outlier
        future to be removed
        '''
        df_s1 = df.shape[0]
        z = np.abs(stats.zscore(df[self.YVAR].values))        
        z_idx = np.where(z < thres)
        #print(np.shape(z_idx))
        #print(z_idx)
        print('\nn outliers : {}\n'.format(df_s1-np.shape(z_idx[0])[0]))
        df = df.iloc[z_idx[0]].reset_index().copy()
        return df



    def _parse_samples(self, dffile, FUNC): 
        '''
        for each file, each unique ID will be analyzed
        using a FUNC
        
        this function will ask if we want to work on raw or smoothed data
        '''


        # activate or inacte remove outlier possibility
        # future : will be removed, experimental use only
        # Currently : deactivated
        if self.Conductance:
            self._turn_on_off_remove_outlier(state=False)
        else:
            self._turn_on_off_remove_outlier(state=False)   

        # for each file, slice each unique ID
        for sample in dffile[self.SAMPLE_ID].unique():
            
            self.sample = sample
            df = dffile.loc[dffile[self.SAMPLE_ID]==sample,:].copy().reset_index()
            if self.remove_outlier:
                df = self._detect_outlier(df=df, thres =THRES)

            # transform time to TRUE date time
            try:
                df['delta_time']
            except:
                print('delta time column is leaking ... computing ...')
                df['TIME_COL2'] = pd.to_datetime(df[self.TIME_COL] , infer_datetime_format=True)  
                # compute time delta between measures
                # WARNING : the points need to be regurlarly sampled with a constant frequency
                df['delta_time'] = (df['TIME_COL2']-df['TIME_COL2'][0])   
                # convert time to minute
                df['delta_time'] = df['delta_time'].dt.total_seconds() / 60 # minutes 

            ################# PUT HERE THE RWC METHOD
            #########################################

            self.Xselected = df['delta_time'].values
            self.yselected = df[self.YVAR].copy().values
            #print(FRAC_P)
            #print(DELTA_MULTI)
            self.Ysmooth = self._smoother(self.Xselected , self.yselected, fr = FRAC_P, delta_multi = DELTA_MULTI)

            df['raw_data'] = df[self.YVAR].copy()
            df['smooth_data'] = self.Ysmooth.copy()
            df['Work_on_smooth'] = 'No'

            if not self.Conductance or self.Conductance:  
                # strange code : but always ask for smoothing
                # future : can be modified (either always allow smoothing and self.conductance is not useful or smoothinf is forbiden for robust differential)
                plt.plot(self.Xselected, self.yselected, linestyle='-', marker='.', label = 'raw data')
                plt.plot(self.Xselected, self.Ysmooth, linestyle='-', marker='.', label = 'smooth')
                plt.title(self.sample)
                plt.ylabel(self.sample)
                plt.legend()    
                # plt.pause(PAUSE_GRAPH/10)
                plt.waitforbuttonpress(0)
                # input()
                plt.close() 

            
                print('''
                Do you want to work on smoothed data ?

                1: Yes
                2: Yes, But I want to adjust smoothing parameters
                3: No            
                ''') 

                what_to_do = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2','3'))
                ########################################################################
                if what_to_do=='1':
                    df[self.YVAR] = self.Ysmooth.copy()
                    df['Work_on_smooth'] = 'yes'
                if what_to_do=='2':
                    while True:          
                        while True:
                            
                            try:
                                _FRAC=0.1
                                FRAC_P2 = float(input('What is the frac value ? (current value : {}) '.format(_FRAC)) or _FRAC)
                                _FRAC = FRAC_P2
                                break
                            except ValueError:
                                print("Oops!  That was no valid number.  Try again...")                    
                        while True:

                            try:
                                _DELTA_MULTI=0.01
                                DELTA_MULTI2= float(input('What is the delta value ? (current value : {}) '.format(_DELTA_MULTI)) or _DELTA_MULTI)
                                _DELTA_MULTI = DELTA_MULTI2
                                break
                            except ValueError:
                                print("Oops!  That was no valid number.  Try again...")

                        self.Ysmooth = self._smoother(self.Xselected , self.yselected, fr = FRAC_P2, delta_multi = DELTA_MULTI2)
                        plt.plot(self.Xselected, self.yselected, linestyle='-', marker='.', label = 'raw data')
                        plt.plot(self.Xselected, self.Ysmooth, linestyle='-', marker='.', label = 'smooth')
                        plt.legend()    
                        # plt.pause(PAUSE_GRAPH/10)
                        plt.waitforbuttonpress(0)
                        # input()
                        plt.close()   
                        print('''
                        Do you want to keep this values?

                        1: Yes
                        2: No            
                        ''')
                        what_to_do = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2'))
                        if what_to_do == '1':
                            break
                        if what_to_do == '2':
                            pass
                    
                    df[self.YVAR] = self.Ysmooth.copy()
                    df['Work_on_smooth'] = 'yes'
        
            FUNC(df)
            #return dfe
         

    def _robust_import(self, elem):

        '''
        try to open a csv file using several methods
        should be relatively robust
        future : robustness could be improved

        use parsefile class        
        '''
        if self.file_or_folder == '2':
            
            if self.method_choice== '2':                
                skip=1
            else:
                skip=0

            try:
                dffile = ParseFile(path = elem, skipr=skip).clean_file()
            except:
                encodi='latin'
                dffile = ParseFile(path = elem, skipr=skip, encod=encodi).clean_file()

            if dffile.shape[1] == 1:
                print('TRU*****************************************')
                separ=';'
                try:
                    dffile = ParseFile(path = elem, sepa=separ, skipr=skip).clean_file()
                except:
                    encodi='latin'
                    dffile = ParseFile(path = elem, skipr=skip, sepa=separ, encod=encodi).clean_file()
        

            uniqueid = dffile[self.SAMPLE_ID].unique()
            print('''
            Unique ID within selected file are: 
            {}
                     
            '''.format(uniqueid))

            #idtoanalyse = input("which one do you want to analyse ?")
            listodidtoanalyse = []   
            count = 0
            

            idtoanalyse = ''               

            while True:
                while ((idtoanalyse not in uniqueid) and (idtoanalyse not in ['exit', 'e'])):
                    idtoanalyse = input("\nwhich ID do you want to analyse ?\nPlease select one among:\n{}\nEnter --exit-- to stop\n\nYour choice: ".format(uniqueid))
                    # print('')

                    # print(idtoanalyse)
                    # print(idtoanalyse not in uniqueid)
                    # print(idtoanalyse not in ['exit', 'e'])
                    # print(((idtoanalyse not in uniqueid) or (idtoanalyse not in ['exit', 'e'])))
                    # print(((idtoanalyse not in uniqueid) and (idtoanalyse not in ['exit', 'e'])))

                # print(idtoanalyse, count)
                if  idtoanalyse in uniqueid:
                    print(' \nAppending : {}'.format(idtoanalyse))
                    listodidtoanalyse.append(idtoanalyse)
                    count += 1
                    idtoanalyse = ''
                       
                elif (idtoanalyse == 'exit' or idtoanalyse == 'e'):
                    if count>0:
                        print(' \nExiting')
                        break
                    else:
                        print('\nYou need to choose at least one ID before')
                        idtoanalyse = ''

               
              
            
            boollistofid = [True if id in listodidtoanalyse else False for id in dffile[self.SAMPLE_ID] ]


            dffile = dffile[boollistofid].copy()
            print('\nselected ID are: {}'.format(dffile[self.SAMPLE_ID].unique()))
                        




        if self.file_or_folder == '1':
            if self.method_choice== '2':
                skip=1
            else:
                skip=0
            try:
                dffile = ParseFile(path = elem, skipr=skip).clean_file()
            except:
                encodi='latin'
                dffile = ParseFile(path = elem, skipr=skip, encod=encodi).clean_file()

            if dffile.shape[1] == 1:
                separ=';'
                try:
                    dffile = ParseFile(path = elem, sepa=separ, skipr=skip).clean_file()
                except:
                    encodi='latin'
                    dffile = ParseFile(path = elem, skipr=skip, sepa=separ, encod=encodi).clean_file()
        return dffile
        

    def _compute_rwc(self, df, nmean = 100,  visualise = True):      

        from matplotlib.patches import Circle, Wedge, Polygon

        rwc_thressup = self.rwc_sup
        rwc_thresinf = self.rwc_inf

        dry = np.mean(df[self.YVAR].values[-int(nmean):])
        saturated = np.mean(df[self.YVAR].values[0])
        rwc = 100*((df[self.YVAR].values-dry)/(saturated-dry))            

        def find_nearest(a, a0):
            "Element in nd array `a` closest to the scalar value `a0`"
            idx = np.abs(a - a0).argmin()
            
            return a.flat[idx]  

        rwc_sup = find_nearest(rwc, rwc_thressup)
        rwc_inf = find_nearest(rwc, rwc_thresinf)  

        print('RWC boundary: [{}% .. {}%]'.format(np.round(rwc_sup,2), np.round(rwc_inf,2)))
        



        def compute_time_lag(df):
            df['TIME_COL2'] = pd.to_datetime(df[self.TIME_COL] , infer_datetime_format=True)  
            # compute time delta between measures
            # WARNING : the points need to be regurlarly sampled with a constant frequency
            df['delta_time'] = (df['TIME_COL2']-df['TIME_COL2'][0])   
            # convert time to minute
            df['delta_time'] = df['delta_time'].dt.total_seconds() / 60 # minutes 

            return df
        
        df = compute_time_lag(df)

        t80 = np.min(df.loc[rwc == rwc_sup, "delta_time"].values)
        t50 = np.max(df.loc[rwc == rwc_inf, "delta_time"].values)

        print('Detected RWC SUP {}% at {} min'.format(rwc_thressup,t80))
        print('Detected RWC INF {}% at {} min'.format(rwc_thresinf,t50))
        

        TITLE = str(df[self.SAMPLE_ID].unique()[0])
        if visualise:

            
            
            fig, ax1 = plt.subplots()

            plt.title(TITLE)
            color = 'tab:blue'
            ax1.set_xlabel('time (min)')
            ax1.set_ylabel(TITLE + '\nWeight (g)', color=color)
            ax1.plot(df['delta_time'], df[self.YVAR], color=color, linestyle='-', marker='.', label = 'data', alpha = 0.5)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylim([0.9*np.min(df[self.YVAR]), 1.1*np.max(df[self.YVAR])])

           

            verts = [[0, 0],[t80, 0], [t80, 1.1*np.max(df[self.YVAR].values)] , [0, 1.1*np.max(df[self.YVAR].values)]]
            poly = Polygon(verts, facecolor='r', alpha = 0.5)
            ax1.add_patch(poly)   

            verts = [[t50, 0],[np.max(df['delta_time'].values), 0], [np.max(df['delta_time'].values), 1.1*np.max(df[self.YVAR].values)], [t50, 1.1*np.max(df[self.YVAR].values)] ]
            poly = Polygon(verts, facecolor='r', alpha = 0.5)
            ax1.add_patch(poly)



            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:green' 
            ax2.set_ylabel('RWC (%)', color=color)  # we already handled the x-label with ax1
            ax2.plot(df['delta_time'],  rwc, color=color, marker='.', label = 'RWC', alpha = 0.5)
            ax2.tick_params(axis='y', labelcolor=color) 




            # Make the shaded region
                        
            
        try:
            self.fig_folder
        except:
            self.fig_folder = 'None'

        if self.fig_folder == 'None':
            starting_name = 'output_fig'
            i = 0
            while True:
                i+=1
                fig_folder = starting_name+'_'+str(i)
                if not os.path.exists(fig_folder):
                    os.makedirs(fig_folder)
                    os.makedirs(fig_folder+'/'+'rmse')
                    os.makedirs(fig_folder+'/'+'diff')
                    os.makedirs(fig_folder+'/'+'rwc')
                    break
            self.fig_folder = fig_folder                

        figname = self.fig_folder + '/' + 'rwc' + '/' + TITLE + '.png'
        plt.savefig(figname, dpi = 420, bbox_inches = 'tight')
        # plt.pause(PAUSE_GRAPH)
        #plt.show()
        plt.waitforbuttonpress(0)
        # input()
        plt.close() 




        print('Slicing df between RWC80 and RWC50')
        # df = df[(rwc < rwc_thressup) & (rwc > rwc_thresinf)].copy()
        df = df[ (df.delta_time.values <= t50) & (df.delta_time.values >= t80)].copy()

        print('min : ', df.delta_time.min())
        print('max : ', df.delta_time.max())

        return df, t80, t50

       


    def change_detection(self):
        '''
        parse all the files within the folder tree

       
        '''

        print('change_detection\n')

    
        self.Conductance=False
        dimfolder = len(self.listOfFiles)
        li_all = []
        for d in np.arange(0,dimfolder):
            print('\n\n\n---------------------------------------------------------------------')
            print(d)
            li = []
            try:
                self.presentfile=self.listOfFiles[d][0]
            except:
                self.presentfile = 'No file'
            
            self.df_rmse = None
            self.df_value = None
            print('parsing list of files from : {}'.format(self.presentfile))

            try:
                temp_name = self.rep_name                
            except:
                self.rep_name = 'None'
            
            if self.rep_name == 'None':
                starting_name = 'output_files'
                i = 0
                while True:
                    i+=1
                    temp_name = starting_name+'_'+str(i)
                    if not os.path.exists(temp_name):
                        os.makedirs(temp_name)
                        break

                self.rep_name = temp_name

            print('Saving to : ', temp_name)
            temp_name += '/'
            

            list_of_df = []
   

            if self.presentfile != 'No file':
                for elem in self.listOfFiles[d]:
                    dffile = self._robust_import(elem)          
                    
                    # fit after rwc removal, if you want to use the full curve set tinf to 0 and tsup to 100
                    dffile, t80, t50 = self._compute_rwc(dffile)
                    
                    # future : do i need to use global var as self.globalscore ... ?
                    self._parse_samples(dffile = dffile, FUNC = self._change_det)
                    temp_df = pd.DataFrame(self.global_score, columns = ['Sample_ID', 'Change_points','slope', 'Rsquared', 'Gmin_mean', 'pack'])
                    temp_df2 = pd.DataFrame(temp_df["pack"].to_list(), columns=['K', 'VPD', 'mean_T', 'mean_RH', 'mean_Patm', 'mean_area'])
                    temp_df = temp_df.drop(columns='pack')
                    
                    # remove the .csv extension from the name
                    temp_folder = os.path.splitext(str(os.path.basename(elem)))[0]

                    if not os.path.exists(temp_name + temp_folder ):
                        os.makedirs(temp_name + temp_folder)
                    # concat df
                    temp_df = pd.concat([temp_df,temp_df2], axis = 1)
                    temp_df['Campaign'] = temp_folder

                    # NEW ##
                    temp_df['fit_mode'] = self.fit_exp_rmse
                    temp_df['transfo_mode'] = self.transfo_rmse

                    temp_df['percentage_rwc_sup']=self.rwc_sup
                    temp_df['percentage_rwc_inf']=self.rwc_inf

                    temp_df['time_rwc_sup']=t80
                    temp_df['time_rwc_inf']=t50

                    # append df to list
                    list_of_df.append(temp_df)
                    temp_df.to_csv(temp_name + temp_folder + '/RMSE_detection_' + str(os.path.basename(elem))) 
                    #pd.concat([temp_df,temp_df2], axis = 1).to_csv('output_files/'+ temp_folder + '/RMSE_detection_' + str(os.path.basename(elem)))                 
                                        
                    self.df_rmse.to_csv(temp_name+ temp_folder + '/RMSE_score_'+str(os.path.basename(elem)))
                    self.df_value.to_csv(temp_name+ temp_folder + '/RMSE_df_complete_'+str(os.path.basename(elem)))
                    self.df_rmse = None
                    self.df_value = None
                    self.global_score = []

            # save the appended df in a global file
            # explode remove the square bracket [] from cells and convert to multiline
            pd.concat(list_of_df).reset_index().explode('Change_points').to_csv(temp_name+'RMSE_df_complete_full.csv')
            pd.concat(list_of_df).reset_index().explode('Change_points').drop_duplicates(subset=['Campaign','index','Sample_ID','slope']).to_csv(temp_name+'RMSE_df_complete_full_No_duplicates.csv')
                
        
    def _plot_tvregdiff(self, _X, _y, _y2, peaks, ax2_Y =r'$Gmin (mmol.m^{-2}.s^{-1})$', ax2_label = 'Gmin' , manual_selection=False, Npoints=1):
        
        fig, ax1 = plt.subplots()
        plt.title(self.sample)
        color = 'tab:blue'
        ax1.set_xlabel('time (min)')
        ax1.set_ylabel(self.sample + '\nWeight (g)', color=color)
        ax1.plot(self.Xselected, self.yselected, color=color, linestyle='-', marker='.', label = 'data')
        ax1.tick_params(axis='y', labelcolor=color)
        color = 'tab:red'
        ax1.plot(self.Xselected, self.Ysmooth, color=color, lw=2, linestyle='-', label = 'smooth')        
        

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:green'       
        
        ax2.set_ylabel(ax2_Y, color=color)  # we already handled the x-label with ax1
        ax2.plot(_X, _y, color=color, marker='.', label = ax2_label)
        ax2.tick_params(axis='y', labelcolor=color)
        
        #ax2.plot(_X[peaks], _y[peaks], "x", markersize=10, mew=4, c = 'red')

        ax3 = ax1.twinx()
        color = 'tab:orange'
        ax3.plot(_X, _y2, color=color, marker='.', label = 'first order derivative ' + ax2_label)
        ax3.tick_params(axis='y', labelcolor=color, 
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)        
        ax3.plot(_X[peaks], _y2[peaks], "x", markersize=10, mew=4, color = 'red', label = 'Peak')
        ax3.vlines(ymin=np.min(_y2),ymax = np.max(_y2),x=_X[peaks], color='red', lw=0.8, linestyle='--')
        ax3.axis('off')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax3.legend(loc='right')
        fig.tight_layout()
        if manual_selection:
            selected_points=fig.ginput(Npoints)


        try:
            self.fig_folder
        except:
            self.fig_folder = 'None'

        if self.fig_folder == 'None':
            starting_name = 'output_fig'
            i = 0
            while True:
                i+=1
                fig_folder = starting_name+'_'+str(i)
                if not os.path.exists(fig_folder):
                    os.makedirs(fig_folder)
                    os.makedirs(fig_folder+'/'+'rmse')
                    os.makedirs(fig_folder+'/'+'diff')
                    os.makedirs(fig_folder+'/'+'rwc')
                    break
            self.fig_folder = fig_folder                

        figname = self.fig_folder + '/' + 'diff' + '/' + self.sample + '.png'
        plt.savefig(figname, dpi = 420, bbox_inches = 'tight')
        # plt.pause(PAUSE_GRAPH)
        #plt.show()
        plt.waitforbuttonpress(0)
        # input()
        plt.close()   
        if manual_selection:
            return selected_points
    
    
    def _compute_gmin(self, df, slope):                

        k= (slope/18.01528)*(1000/60) #ici c'est en minutes (60*60*24)

        #Calcul VPD en kpa (Patm = 101.325 kPa)
        VPD =0.1*((6.13753*np.exp((17.966*(df[self.T].values)/((df[self.T].values)+247.15)))) - ((df[self.RH].values)/100*(6.13753*np.exp((17.966*(df[self.T].values)/((df[self.T].values)+247.15)))))) 

        #calcul gmin mmol.s
        gmin_ = -k * (df[self.PATM].values)/VPD

        #calcul gmin en mmol.m-2.s-1
        gmin = gmin_ / (df[self.AREA].values)

        #print('gmin : ', gmin)
        try:
            print('gmin mean last 100 values: ', np.mean(gmin[-100:]))
        except:
            print('gmin mean last 10 values: ', np.mean(gmin[-10:]))
        try:
            print('slope mean last 100 values: ', np.mean(slope[-100:]))
        except:
            print('slope mean last 10 values: ', np.mean(slope[-10:]))
        return gmin, k, VPD

    def _tvregdiff(self,df):

        _X = df['delta_time'].copy().values
        _y = df[self.YVAR].copy().values

        dX = _X[1] - _X[0]
        if len(_X)<1000:
            SCALE = 'small'
            PRECOND = False
        else:
            SCALE = 'large'
            PRECOND = True

        if len(_X)<200:   # MAYBE HYPERPARAMETERS CAN BE DEFINED OTHERLY
            DIV_ALPH = 10 # 1000
            DIV_ALPH2 = 1000 # 1000
            DIST = 10
            PROM = 20 #10
            #EP = EP
            EP2 = self.EP            
        else:
            DIV_ALPH = 1 #10
            DIV_ALPH2= 50
            DIST = 50 #200
            PROM = 3#4
            #EP = EP
            EP2 = self.EP*1

        dYdX = TVRegDiff(data=_y ,itern=self.ITERN, 
                        alph=self.ALPH/DIV_ALPH, dx=dX, 
                        ep=self.EP,
                        scale=SCALE ,
                        plotflag=False, 
                        precondflag=PRECOND,
                        diffkernel=self.KERNEL,
                        u0=np.append([0],np.diff(_y)),
                        cgtol = 1e-4)   

        gmin, k, VPD = self._compute_gmin(slope = dYdX, df = df)             

        
        # _change_det(self, df, COL_Y='standard')
        df['gmin'] = gmin
        peaks, X_peaks, Xpeaks_int = self._change_det(df, COL_Y='gmin', mode = 'diff') # return idx, Xidx, Xidx_int 
        
        #####################################################################################"
        # 
        _ALPH = self.ALPH/DIV_ALPH
        _ALPH2=self.ALPH/DIV_ALPH2
        _EP=self.EP
        _EP2=EP2

        print('''
            Do you want to keep this parameters for conductance computation ?

            1: Yes or exit
            2: No, I want to adjust regularization parameters
            3: No, I want to select peaks manually (- deactivated -)   
                    
            ''') 

        what_to_do = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2')) #option '3' removed
        ########################################################################
        # keep the firt detected peak to avoid slice issues
        peaks_0 = peaks
        while True:
            if what_to_do=='1':
                break
            if what_to_do=='3':
                while True:
                    try:                        
                        _Npoints = int(self._get_valid_input('How many points do you want to select ? ',('1', '2')) or 1)                
                        break
                    except ValueError:
                        print("Oops!  That was no valid number.  Try again...")
                
                # sel_p = self._plot_tvregdiff(_X=_X[:], _y=gmin[:], _y2 = ddGmin, peaks=peaks_0, manual_selection=True, Npoints=_Npoints)  
                sel_p = self._plot_tvregdiff(_X=_X[:], _y=gmin[:], _y2 = gmin, peaks=peaks_0, manual_selection=True, Npoints=_Npoints)  

                peaks = [str(np.round(i[0],3)) for i in sel_p]
                #peaks = '['+peaks+']'

                print('Selected points at time : ', ' '.join(map(str,peaks)))


                print('''
                        Do you want to keep this parameters for conductance computation ?

                        1: Yes or exit
                        2: No, I want to adjust regularization parameters
                        3: No, I want to select peaks manually       
                             
                        ''') 

                what_to_do = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2', '3'))

            if what_to_do=='2':
                
                print('''                
                    Alpha : Higher values increase regularization strenght and improve conditioning         
                    Epsilon : Parameter for avoiding division by zero smaller values give more accurate results with sharper jumps
                    ''')
                #while True:          
                while True:                        
                    try:                        
                        _ALPH = float(input('What is the value for alpha? (current value : {}) '.format(_ALPH)) or _ALPH)
                        break
                    except ValueError:
                        print("Oops!  That was no valid number.  Try again...")                  

                while True:
                    try:                        
                        _EP= float(input('What is the value for epsilon ? (current value : {}) '.format(_EP))or _EP)
                        break
                    except ValueError:
                        print("Oops!  That was no valid number.  Try again...")

                dYdX = TVRegDiff(data=_y ,itern=self.ITERN, 
                    alph=_ALPH, dx=dX, 
                    ep=_EP,
                    scale=SCALE ,
                    plotflag=False, 
                    precondflag=PRECOND,
                    diffkernel=self.KERNEL,
                    u0=np.append([0],np.diff(_y)),
                    cgtol = 1e-4)        
       
                gmin, k, VPD = self._compute_gmin(slope = dYdX, df = df)
                df['gmin'] = gmin
                peaks, X_peaks, Xpeaks_int = self._change_det(df, COL_Y='gmin', mode = 'diff') # re

                print('''
                    Do you want to keep this parameters for conductance computation ?

                    1: Yes or exit
                    2: No, I want to adjust regularization parameters
                    3: Yes, but I want to select peaks manually  

                    ''') 

                what_to_do = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2', '3'))

        # ###################################################################################"
        df['raw_slope'] = dYdX
        df['gmin'] = gmin
        
        #df['d_gmin'] =  ['NA']+ddGmin.tolist()
        df['d_gmin'] =  'None'

        #print(df)
        if len(peaks)>0:
            try:
                # df['Peaks'] = np.array_str(_X[peaks])
                df['Peaks'] = np.array_str(X_peaks)
            except:                
                # df['Peaks'] = ' '.join(map(str,peaks))
                df['Peaks'] = ' '.join(map(str,X_peaks))

        else:
            df['Peaks'] = 'NoPeak'
        assert self.df_save.columns.size ==  df.columns.size, 'size differs between df_save & df'
        self.df_save.columns = df.columns
        self.df_save = pd.concat([self.df_save, df], axis = 0, ignore_index = True)

        return df

    def robust_differential(self):    

        print('''
         WARNING : algorithm convergence time may be long
        ''')
        time.sleep(0.5)
        self.Conductance = True
        dimfolder = len(self.listOfFiles)
        li_all = []
        # future add a way to check the length of the columns section
        self.df_save = pd.DataFrame(columns = range(0,18))
        self.df_value = None
        self.df_rmse = None
        for d in np.arange(0,dimfolder):
            print('\n\n\n---------------------------------------------------------------------')
            print(d)
            li = []
            try:
                self.presentfile=self.listOfFiles[d][0]
            except:
                self.presentfile = 'No file'
            
            print('parsing list of files from : {}'.format(self.presentfile))
            if self.presentfile != 'No file':
                for elem in self.listOfFiles[d]:
                    dffile = self._robust_import(elem)
                    self._parse_samples(dffile = dffile, FUNC = self._tvregdiff)
        
        
        # NEW ##
        self.df_save['fit_mode'] = self.fit_exp_diff
        self.df_save['transfo_mode'] = self.transfo_diff
        try:
            self.df_save.to_csv(self.rep_name + '/gmin.csv')
        except:
            print('rep not found creating rep')
            self.rep_name = 'None'

        if self.rep_name == 'None':
            starting_name = 'output_files'
            i = 0
            while True:
                i+=1
                temp_name = starting_name+'_'+str(i)
                if not os.path.exists(temp_name):
                    os.makedirs(temp_name)
                    break

            self.rep_name = temp_name
            print('Saving to : ', temp_name)
            temp_name += '/'
            self.df_save.to_csv(temp_name + 'gmin.csv')



        



            