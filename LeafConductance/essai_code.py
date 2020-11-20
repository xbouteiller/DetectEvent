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
print('------------------------------------------------------------------------')
print('---------------                                    ---------------------')
print('---------------            LeafConductance         ---------------------')
print('---------------                  V0.1              ---------------------')
print('---------------                                    ---------------------')
print('------------------------------------------------------------------------')
time.sleep(0.5)

num_col = ['weight_g','T_C','RH', 'Patm']
group_col=['campaign', 'sample_ID', 'Treatment', 'Operator']
date=['date_time']


# GLOBAL VARIABLES
SEP = ','
TIME_COL = 'date_time'
SAMPLE_ID = 'sample_ID'
YVAR = 'weight_g'
WIND_DIV = 8
LAG_DIV = WIND_DIV * 45
BOUND = 'NotSet'
FRAC_P = 0.1
DELTA_MULTI = 0.01
PAUSE_GRAPH = 8

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
        '''
        import re
        import pandas as pd
        import numpy as np


        return self.file



class ParseTreeFolder():


    def _get_valid_input(self, input_string, valid_options):
        input_string += "({}) ".format(", ".join(valid_options))
        response = input(input_string)
        while response.lower() not in valid_options:
            response = input(input_string)
        return response


    def __init__(self):
        import time
        # super().__init__()
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename, askdirectory

        self.global_score = []

        print('''
        WELCOME TO LEAFCONDUCTANCE

        Press Enter to continue ...
        
        ''')
        self.file_or_folder = '1'
        input('')
        # self.file_or_folder = self._get_valid_input('Type 1 to start : ', ('1'))
        if self.file_or_folder== '1':
            ################################################### REACTIVATE
            #Tk().withdraw()
            #folder = askdirectory(title='What is the root folder that you want to parse ?')
            folder = '/home/xavier/Documents/development/DetectEvent/data'
            #####################################################""
            self.path = folder
            print('\n\n\nroot path is {}'.format(self.path))
            ################################################### REACTIVATE
            # print('''
            # which method do you want to use for detecting cavisoft files ?

            # 1: Detect all CSV files 
            # 2: Detect string 'CONDUCTANCE' string in the first row
            # ''')
            # self.method_choice = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2'))
            self.method_choice = '2' 
            ################################################### REACTIVATE
            print('\n\n\nfile 1 path is {}'.format(self.path))


        self.choices = {
        "1": self.change_detection#,
        # "2": self.compute_conductance,
        # "3": self.extract_strings,
        # "4": self.erase,
        # "5": self.extract_strings_and_nums
        }


    def _listdir_fullpath(self, p, s):
        import os
        import re
        d=os.path.join(p, s)
        return [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.csv')]

    def _detect_cavisoft(self, p, s):
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


        if self.file_or_folder=='1':
            file_root=[]
            self.listOfFiles = []


            if self.method_choice == '1':
                try:
                    file_root = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith('.csv')]
                    self.listOfFiles.append(file_root)
                    print(file_root)
                except:
                    print('no file detected within root directory')
                    pass

                try:
                    for pa, subdirs, files in os.walk(self.path):
                        for s in subdirs:
                            self.listOfFiles.append(self._listdir_fullpath(p=pa, s=s))
                except:
                    print('no file detected within childs directory')
                    pass

            if self.method_choice == '2':
                try:
                    file_root = [os.path.join(self.path, f) for f in os.listdir(self.path) if\
                    f.endswith('.csv') and (re.search(r'conductance', pd.read_csv(os.path.join(self.path, f),sep=SEP,nrows=0).columns[0].lower()) )]
                    self.listOfFiles.append(file_root)


                    print(file_root)
                except:
                    print('no file detected within root directory')
                    pass

                try:
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

            time.sleep(4)

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
        List of actions

        1. Detect changes in curve (RMSE approach)
        2. Compute conductance (robust differential approach)
        """)

    def run(self):
        '''Display the menu and respond to choices.'''

        self.display_menu()
        choice = input("Enter an option: ")
        action = self.choices.get(choice)
        if action:
            action()
        else:
            print("{0} is not a valid choice".format(choice))
            self.run()


    def _min_max(self, X): 
        X_scaled = (X-np.min(X)) / (np.max(X) -  np.min(X))
        return X_scaled

    def _RMSE(self, Ypred, Yreal):
        rmse = np.sqrt(np.sum(np.square(Ypred-Yreal))/np.shape(Ypred)[0])
        return rmse

    def _fit_and_pred(self,X, y, mode, *args):
        if mode == 'linear':
            Xarr = np.array(X).reshape(-1,1)
            yarr = np.array(y).reshape(-1,1)
            reg = LinearRegression().fit(Xarr, yarr)
            pred = reg.predict(Xarr)
        if mode == 'exp':
            Xarr = np.array(X).reshape(-1)
            yarr = np.array(y).reshape(-1)
            reg = curve_fit(self._func, Xarr, yarr, bounds=args[0])[0]                                         
            A, B = reg     
            pred = A * np.exp(-B * Xarr)
        rmse = self._RMSE(pred, yarr)
        return rmse
    
    def _sliding_window_pred(self, X, y, window, lag, mode, b=BOUND):
        Xend = np.shape(X)[0]
        Xmax = np.shape(X)[0]-lag-1
        start = np.arange(window, Xend-window, lag)

        if mode == 'linear':
            mean_start = X[[int(Xend-s) for s in start]]
            score = [self._fit_and_pred(X[Xend-s:Xend], y[Xend-s:Xend], mode) 
                    for s in start]    #[::-1]
        
        if mode == 'exp':
            mean_start = X[start]
            if BOUND == 'NotSet':
                try:
                    reg = self._detect_b( X[lag:lag+(window*4)], y[lag:lag+(window*4)], mode)
                    Aa, Bb = reg 
                    #bound = ([Aa-0.1*Aa,Bb/10],[Aa+0.1*Aa, Bb*10]) # TO DO CHECK IF IT IS CORRECT
                    bound = ([Aa-0.5*Aa,Bb/100],[Aa+0.5*Aa, Bb*100])
                except:
                    bound = ([0,1/1000000],[100, 1/100])
            else:
                bound=BOUND

            try:
                score = [self._fit_and_pred(X[0:s], y[0:s], mode, bound) 
                        for s in start]
            except:
                raise Exception('Failed to fit Exponential curve')

        score = self._min_max(score)
        print('{} score'.format(mode), score)
        return score, mean_start


    def _func(self, x, a, b):
        return a * np.exp(-b * x) 

    def _func_lin(self, x, a, b):
        return a * x + b 
    
    def _detect_b(self, X, y, mode):
        Xarr = np.array(X).reshape(-1)
        yarr = np.array(y).reshape(-1)
        if mode == 'exp':
            reg = curve_fit(self._func, Xarr, yarr)[0]
        return reg

    def _dcross(self, Yl, Ye):
        idx = np.argwhere(np.diff(np.sign(Yl - Ye))).flatten() 
        return idx

    def _detect_crossing_int(self, Yexp, Ylin, Xl, Xe):
        Ylin=np.array(Ylin)
        Yexp=np.array(Yexp)
        Xl=np.array(Xl)  
        Xe=np.array(Xe) 
        idx = self._dcross(Ylin[::-1], Yexp)
        Xidx=Xe[idx]    
        idx_int = [[i, i+1] for i in idx]
        Xidx_int = [[Xe[i], Xe[i+1] ]for i in idx]

        Yidx=self.Ysmooth[self.Xselected == Xidx]


        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('time (min)')
        ax1.set_ylabel(self.sample, color=color)
        ax1.plot(self.Xselected, self.yselected, color=color, linestyle='-', marker='.', label = 'data')
        ax1.tick_params(axis='y', labelcolor=color)
        color = 'tab:red'
        ax1.plot(self.Xselected, self.Ysmooth, color=color, lw=2, linestyle='-', label = 'smooth')        
        ax1.plot(Xidx, Yidx, 'ro', markersize=8)
        ax1.hlines(xmin=0,xmax=self.Xselected[-1],y=Yidx, color='red', lw=0.8, linestyle='--')
        ax1.vlines(ymin=np.min(self.yselected),ymax = np.max(self.yselected),x=Xidx, color='red', lw=0.8, linestyle='--')
        ax1.legend()

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        ax2.set_ylabel('RMSE')  # we already handled the x-label with ax1
        ax2.plot(Xl, Ylin, color=color,  marker='.', label = 'RMSE lin')
        #ax2.tick_params(axis='y', labelcolor=color)
        color = 'tab:orange'
        ax2.set_ylabel('RMSE', color=color)  # we already handled the x-label with ax1
        ax2.plot(Xe, Yexp, color=color, marker='.', label = 'RMSE exp')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend()
        fig.tight_layout()
        # plt.pause(PAUSE_GRAPH)
        plt.waitforbuttonpress(0)
        # input()
        plt.close()   

        print('\nInterval method')
        for i in np.arange(0,len(idx)):
            print('detected changes between times : {} - {}'.format(Xidx_int[i][0], Xidx_int[i][1]))
        
        print('''
            Do you want to keep the crossing value ?

            1: Yes, save
            2: No, discard
            3. Select value manually on graph (WORK IN PROGRESS)
            ''')
        what_to_do = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2'))

        if what_to_do=='2':
            self.global_score.append([self.sample,'Discarded'])
        if what_to_do=='1':
            self.global_score.append([self.sample, Xidx])

        print('gs',self.global_score)

        return idx, Xidx, Xidx_int 
    
    def _change_det(self, df):

        if df.shape[0] < 100:
            _wind = int(df.shape[0]/6)
            _lag = 1# int(df.shape[0]/4)
        else:
            _wind = int(df.shape[0]/WIND_DIV)
            _lag = int(df.shape[0]/LAG_DIV)
        _X = df['delta_time'].copy().values
        _y = df[YVAR].copy().values

        score_l, mean_start_l = self._sliding_window_pred(_X, _y, window=_wind, lag=_lag, mode = 'linear')
        score_e, mean_start_e = self._sliding_window_pred(_X, _y, window=_wind, lag=_lag, mode = 'exp')
  
        idx, Xidx, Xidx_int = self._detect_crossing_int(Ylin=score_l, Yexp=score_e, Xl= mean_start_l, Xe= mean_start_e) #Yexp, Ylin, Xl, Xe

    def _smoother(self, ex, end, fr, delta_multi):
        delt = delta_multi * ex.shape[0]
        Ysmooth = lowess(exog = ex, endog = end, frac = fr, delta = delt, return_sorted = False)
        return Ysmooth

    def _parse_samples(self, dffile, FUNC): 
        import pandas as  pd
        import numpy as np       

        for sample in dffile[SAMPLE_ID].unique():
            self.sample = sample
            df = dffile.loc[dffile[SAMPLE_ID]==sample,:].copy().reset_index()
            df['TIME_COL2'] = pd.to_datetime(df[TIME_COL] , infer_datetime_format=True)  
            df['delta_time'] = (df['TIME_COL2']-df['TIME_COL2'][0])   
            df['delta_time'] = df['delta_time'].dt.total_seconds() / 60 # minutes 

            self.Xselected = df['delta_time'].values
            self.yselected = df[YVAR].copy().values
            print(FRAC_P)
            print(DELTA_MULTI)
            self.Ysmooth = self._smoother(self.Xselected , self.yselected, fr = FRAC_P, delta_multi = DELTA_MULTI)

            plt.plot(self.Xselected, self.yselected, linestyle='-', marker='.', label = 'raw data')
            plt.plot(self.Xselected, self.Ysmooth, linestyle='-', marker='.', label = 'smooth')
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
                df[YVAR] = self.Ysmooth.copy()
            if what_to_do=='2':
                while True:          
                    while True:
                        
                        try:
                            _FRAC=0.1
                            FRAC_P2 = float(input('What is the frac value ? (current value : {}) '.format(_FRAC)))
                            _FRAC = FRAC_P2
                            break
                        except ValueError:
                            print("Oops!  That was no valid number.  Try again...")                    
                    while True:

                        try:
                            _DELTA_MULTI=0.01
                            DELTA_MULTI2= float(input('What is the delta value ? (current value : {}) '.format(_DELTA_MULTI)))
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
                df[YVAR] = self.Ysmooth.copy()
    
            FUNC(df)
         

    def _robust_import(self, elem):
        if self.file_or_folder== '1':
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
        
    def change_detection(self):
        print('change_detection\n')

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
            print('parsing list of files from : {}'.format(self.presentfile))

            if self.presentfile != 'No file':
                for elem in self.listOfFiles[d]:
                    dffile = self._robust_import(elem)

                    # if self.file_or_folder== '1':
                    #     skip=1
                    # else:
                    #     skip=0
                    # try:
                    #     dffile = ParseFile(path = elem, skipr=skip).clean_file()
                    # except:
                    #     encodi='latin'
                    #     dffile = ParseFile(path = elem, skipr=skip, encod=encodi).clean_file()

                    # if dffile.shape[1] == 1:
                    #     separ=';'
                    #     try:
                    #         dffile = ParseFile(path = elem, sepa=separ, skipr=skip).clean_file()
                    #     except:
                    #         encodi='latin'
                    #         dffile = ParseFile(path = elem, skipr=skip, sepa=separ, encod=encodi).clean_file()


                    self._parse_samples(dffile = dffile, FUNC = self._change_det)
                    pd.DataFrame(self.global_score, columns = ['Sample_ID', 'Wind_of_Change']).to_csv('global_score.csv')



  


            