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
print('---------------                  V0.2              ---------------------')
print('---------------                                    ---------------------')
print('------------------------------------------------------------------------')
time.sleep(0.5)

num_col = ['weight_g','T_C','RH', 'Patm', 'Area_m2']
group_col=['campaign', 'sample_ID', 'Treatment', 'Operator']
date=['date_time']


# GLOBAL VARIABLES
SEP = ','
TIME_COL = 'date_time'
SAMPLE_ID = 'sample_ID'
YVAR = 'weight_g'
T = 'T_C'
RH = 'RH'
PATM = 'Patm'
AREA = 'Area_m2'

WIND_DIV = 8
LAG_DIV = WIND_DIV * 200
BOUND = 'NotSet'
FRAC_P = 0.1
DELTA_MULTI = 0.01
PAUSE_GRAPH = 8

ITERN=4000
ALPH=1000
EP=1e-9
KERNEL='abs'#'abs'
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
        self.Conductance = False
        self.remove_outlier = False

        print('''
        WELCOME TO LEAFCONDUCTANCE

        Press Enter to continue ...
        
        ''')
        self.file_or_folder = '1'
        input('')
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


        self.choices = {
        "1": self.change_detection,
        "2": self.robust_differential,
        "3": self._quit,
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
                    reg = self._detect_b( X[lag:lag+(window*2)], y[lag:lag+(window*2)], mode)
                    Aa, Bb = reg                     
                    bound = ([Aa-0.015*Aa,Bb/1.05],[Aa+0.015*Aa, Bb*1.05])
                except:
                    bound = ([0,1/1000000],[100, 1/100])
            else:
                bound=BOUND
            print('bound : ', bound)
            try:
                score = [self._fit_and_pred(X[0:s], y[0:s], mode, bound) 
                        for s in start]
            except:
                raise Exception('Failed to fit Exponential curve')

        score = self._min_max(score)
        #print('{} score'.format(mode), score)
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

    def _compute_slope(self, Xidx1, interval = False, *args, **kwargs):         

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
        
        if t2 is None:
            df = df[df['delta_time']>= t1].copy()
        if t2 is not None:            
            df = df[(df['delta_time']>= t1) & (df['delta_time']<= t2)].copy()

        k= (slope/18.01528)*(1000/60) #ici c'est en minutes (60*60*24)

        #Calcul VPD en kpa (Patm = 101.325 kPa)
        VPD =0.1*((6.13753*np.exp((17.966*np.mean(df[T].values)/(np.mean(df[T].values)+247.15)))) - (np.mean(df[RH].values)/100*(6.13753*np.exp((17.966*np.mean(df[T].values)/(np.mean(df[T].values)+247.15)))))) 

        #calcul gmin mmol.s
        gmin_ = -k * np.mean(df[PATM].values)/VPD

        #calcul gmin en mmol.m-2.s-1
        gmin = gmin_ / np.mean(df[AREA].values)

        print('gmin_mean: ', gmin)

        return gmin, [k, VPD, np.mean(df[T].values), np.mean(df[RH].values), np.mean(df[PATM].values), np.mean(df[AREA].values)]



    def _detect_crossing_int(self, Yexp, Ylin, Xl, Xe, df):
        gmin_mean=''
        list_of_param=['', '', '', '', '', '']

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
        ax1.plot(self.Xselected, self.yselected, color=color, linestyle='-', marker='.', label = 'Weight (g)')
        ax1.tick_params(axis='y', labelcolor=color)
        color = 'tab:red'
        ax1.plot(self.Xselected, self.Ysmooth, color=color, lw=2, linestyle='-', label = 'smooth')        
        ax1.plot(Xidx, Yidx, 'ro', markersize=8)
        ax1.hlines(xmin=0,xmax=self.Xselected[-1],y=Yidx, color='red', lw=0.8, linestyle='--')
        ax1.vlines(ymin=np.min(self.yselected),ymax = np.max(self.yselected),x=Xidx, color='red', lw=0.8, linestyle='--')
        ax1.legend(loc='upper right')

        if len(Xidx)==1:
            slope, intercept, rsquared, fitted_values, Xreg = self._compute_slope(Xidx1=Xidx)
            ax1.plot(Xreg, fitted_values, c = colors['black'], lw = 2)
            gmin_mean, list_of_param = self._compute_gmin_mean(df=df, slope=slope, t1=Xidx[0], t2 = None)
        else:
            print('more than 1 crossing point were detected')


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
            3. Select values manually on graph
            ''')
        what_to_do = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2','3'))

        if what_to_do=='2':
            self.global_score.append([self.sample,'Discarded','Discarded','Discarded','Discarded'])
        if what_to_do=='1':
            self.global_score.append([self.sample, Xidx, slope, rsquared, gmin_mean, list_of_param])
        if what_to_do=='3':
            while True:
                try:                        
                    _Npoints = int(input('How many points do you want to select ? ') or 1)                
                    break
                except ValueError:
                    print("Oops!  That was no valid number.  Try again...")

            First_pass = 0
            while First_pass < 2:
                fig, ax1 = plt.subplots()

                color = 'tab:blue'
                ax1.set_xlabel('time (min)')
                ax1.set_ylabel(self.sample, color=color)
                ax1.plot(self.Xselected, self.yselected, color=color, linestyle='-', marker='.', label = 'Weight (g)')
                ax1.tick_params(axis='y', labelcolor=color)
                color = 'tab:red'
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
                
                plt.waitforbuttonpress(0)      
                plt.close()
                First_pass+=1

           
            print('\nSelected points at time : ', ' '.join([str(i[0]) for i in selected_points ]))
            print('\n')
            self.global_score.append([self.sample, [i[0] for i in selected_points ], slope, rsquared, gmin_mean, list_of_param])  


        print('gs',self.global_score)

        return idx, Xidx, Xidx_int 
    
    def _change_det(self, df, COL_Y='standard'):

        if df.shape[0] < 100:
            _wind = int(df.shape[0]/6)
            _lag = 1# int(df.shape[0]/4)
        else:
            _wind = max(int(df.shape[0]/WIND_DIV),int(1))
            _lag = max(int(df.shape[0]/LAG_DIV),int(1))
        _X = df['delta_time'].copy().values
        if COL_Y == 'standard':
            _y = df[YVAR].copy().values
        else:
             _y = df[COL_Y].copy().values

        #print(df.head())
        #input()

        if self.df_value is None:
            self.df_value = pd.DataFrame(columns = df.columns)
        
        self.df_value = pd.concat([self.df_value, df], axis = 0, ignore_index = True)


        score_l, mean_start_l = self._sliding_window_pred(_X, _y, window=_wind, lag=_lag, mode = 'linear')
        score_e, mean_start_e = self._sliding_window_pred(_X, _y, window=_wind, lag=_lag, mode = 'exp')

        if self.df_rmse is None:
            self.df_rmse = pd.DataFrame(columns = ['Sample', 'RMSE_lin', 'Time_lin', 'RMSE_exp', 'Time_exp'])        
        
        df_temp_rmse = pd.DataFrame({'Sample':self.sample, 'RMSE_lin':score_l, 'Time_lin':mean_start_l, 'RMSE_exp':score_e, 'Time_exp':mean_start_e})        
        self.df_rmse = pd.concat([self.df_rmse, df_temp_rmse], axis = 0, ignore_index = True)

        try:
            idx, Xidx, Xidx_int = self._detect_crossing_int(Ylin=score_l, Yexp=score_e, Xl= mean_start_l, Xe= mean_start_e, df = df) #Yexp, Ylin, Xl, Xe
        except:
            print('detect crossing failed, probable cause is that more than 1 crossing were detected')
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
        df_s1 = df.shape[0]
        z = np.abs(stats.zscore(df[YVAR].values))        
        z_idx = np.where(z < thres)
        #print(np.shape(z_idx))
        #print(z_idx)
        print('\nn outliers : {}\n'.format(df_s1-np.shape(z_idx[0])[0]))
        df = df.iloc[z_idx[0]].reset_index().copy()
        return df



    def _parse_samples(self, dffile, FUNC):  

        if self.Conductance:
            self._turn_on_off_remove_outlier(state=True)
        else:
            self._turn_on_off_remove_outlier(state=False)   

        for sample in dffile[SAMPLE_ID].unique():
            
            self.sample = sample
            df = dffile.loc[dffile[SAMPLE_ID]==sample,:].copy().reset_index()
            if self.remove_outlier:
                df = self._detect_outlier(df=df, thres =THRES)

            df['TIME_COL2'] = pd.to_datetime(df[TIME_COL] , infer_datetime_format=True)  
            df['delta_time'] = (df['TIME_COL2']-df['TIME_COL2'][0])   
            df['delta_time'] = df['delta_time'].dt.total_seconds() / 60 # minutes 

            self.Xselected = df['delta_time'].values
            self.yselected = df[YVAR].copy().values
            #print(FRAC_P)
            #print(DELTA_MULTI)
            self.Ysmooth = self._smoother(self.Xselected , self.yselected, fr = FRAC_P, delta_multi = DELTA_MULTI)

            df['raw_data'] = df[YVAR].copy()
            df['smooth_data'] = self.Ysmooth.copy()
            df['Work_on_smooth'] = 'No'

            if not self.Conductance:  
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
                    df['Work_on_smooth'] = 'yes'
                if what_to_do=='2':
                    while True:          
                        while True:
                            
                            try:
                                _FRAC=0.1
                                FRAC_P2 = float(input('What is the frac value ? (current value : {}) '.format(_FRAC)) or FRAC_P2)
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
                    
                    df[YVAR] = self.Ysmooth.copy()
                    df['Work_on_smooth'] = 'yes'
        
            FUNC(df)
            #return dfe
         

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
            if not os.path.exists('output_files'):
                os.makedirs('output_files')

            if self.presentfile != 'No file':
                for elem in self.listOfFiles[d]:
                    dffile = self._robust_import(elem)                    
                    
                    self._parse_samples(dffile = dffile, FUNC = self._change_det)
                    temp_df = pd.DataFrame(self.global_score, columns = ['Sample_ID', 'Change_points','slope', 'Rsquared', 'Gmin_mean', 'pack'])
                    temp_df2 = pd.DataFrame(temp_df["pack"].to_list(), columns=['K', 'VPD', 'mean_T', 'mean_RH', 'mean_Patm', 'mean_area'])
                    temp_df = temp_df.drop(columns='pack')

                    pd.concat([temp_df,temp_df2], axis = 1).to_csv('output_files/RMSE_detection_'+str(os.path.basename(elem)))                   
                    
                    
                    self.df_rmse.to_csv('output_files/RMSE_score_'+str(os.path.basename(elem)))
                    self.df_value.to_csv('output_files/RMSE_df_complete_'+str(os.path.basename(elem)))
                    self.df_rmse = None
                    self.df_value = None
                    self.global_score = []

            # pd.DataFrame(self.global_score, columns = ['Sample_ID', 'Change_points','slope', 'Rsquared']).to_csv('RMSE_detection_'+str(os.path.basename(elem))+'.csv')
            # self.df_rmse.to_csv('RMSE_detection_'+str(os.path.basename(elem))+'.csv')
            # self.df_value.to_csv('RMSE_detection_'+str(os.path.basename(elem))+'.csv')
            # self.df_rmse = None
            # self.df_value = None
        
        
    def _plot_tvregdiff(self, _X, _y, _y2, peaks, ax2_Y =r'$Gmin (mmol.m^{-2}.s^{-1})$', ax2_label = 'Gmin' , manual_selection=False, Npoints=1):
        
        fig, ax1 = plt.subplots()
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
        # plt.pause(PAUSE_GRAPH)
        #plt.show()
        plt.waitforbuttonpress(0)
        # input()
        plt.close()   
        if manual_selection:
            return selected_points
    
    
    def _tvregdiff(self,df):

        _X = df['delta_time'].copy().values
        _y = df[YVAR].copy().values

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
            EP2 = EP            
        else:
            DIV_ALPH = 1 #10
            DIV_ALPH2= 50
            DIST = 50 #200
            PROM = 3#4
            #EP = EP
            EP2 = EP*1

        dYdX = TVRegDiff(data=_y ,itern=ITERN, 
                        alph=ALPH/DIV_ALPH, dx=dX, 
                        ep=EP,
                        scale=SCALE ,
                        plotflag=False, 
                        precondflag=PRECOND,
                        diffkernel=KERNEL,
                        u0=np.append([0],np.diff(_y)),
                        cgtol = 1e-4)    

        
        #Conversion de la pente (en g/jour) en mmol/s
        k= (dYdX/18.01528)*(1000/60) #ici c'est en minutes (60*60*24)

        #Calcul VPD en kpa (Patm = 101.325 kPa)
        VPD =0.1*((6.13753*np.exp((17.966*df[T].values/(df[T].values+247.15)))) - (df[RH].values/100*(6.13753*np.exp((17.966*df[T].values/(df[T].values+247.15)))))) 

        #calcul gmin mmol.s
        gmin_ = -k * df[PATM].values/VPD

        #calcul gmin en mmol.m-2.s-1
        gmin = gmin_ / df[AREA].values
        
        #df['gmin'] = gmin
             
        dGmin = TVRegDiff(data=gmin ,itern=ITERN, 
                        alph=ALPH/DIV_ALPH2, dx=dX, 
                        ep=EP2,
                        scale=SCALE ,
                        plotflag=False, 
                        precondflag=PRECOND,
                        diffkernel=KERNEL,
                        u0=np.append([0],np.diff(gmin)),
                        cgtol = 1e-4)
     
        ddGmin = dGmin
        
        peaks, _ = find_peaks(ddGmin, distance=DIST, prominence = np.max(ddGmin)/PROM)
        peaks2, _ = find_peaks(-ddGmin, distance=DIST, prominence = np.max(ddGmin)/PROM)
        peaks = np.concatenate((peaks, peaks2), axis=None)
        
        self._plot_tvregdiff(_X=_X[:], _y=gmin[:], _y2 = ddGmin, peaks=peaks)   

        
        #####################################################################################"
        # 
        _ALPH = ALPH/DIV_ALPH
        _ALPH2=ALPH/DIV_ALPH2
        _EP=EP
        _EP2=EP2

        print('''
            Do you want to work on keep this parameters for conductance computation ?

            1: Yes or exit
            2: No, I want to adjust regularization parameters
            3: No, I want to select peaks manually           
                    
            ''') 

        what_to_do = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2', '3'))
        ########################################################################
        while True:
            if what_to_do=='1':
                break
            if what_to_do=='3':
                while True:
                    try:                        
                        _Npoints = int(input('How many points do you want to select ? ') or 1)                
                        break
                    except ValueError:
                        print("Oops!  That was no valid number.  Try again...")
                sel_p = self._plot_tvregdiff(_X=_X[:], _y=gmin[:], _y2 = ddGmin, peaks=peaks, manual_selection=True, Npoints=_Npoints)  
                peaks = [str(np.round(i[0],3)) for i in sel_p]
                #peaks = '['+peaks+']'

                print('Selected points at time : ', ' '.join(map(str,peaks)))


                print('''
                        Do you want to work on keep this parameters for conductance computation ?

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
                while True:
                    try:                    
                        _ALPH2= float(input('What is the value for alpha for the derivation ? (current value : {}) '.format(_ALPH2)) or _ALPH2)
                        break
                    except ValueError:
                        print("Oops!  That was no valid number.  Try again...")
                while True:
                    try:
                        
                        _EP2= float(input('What is the value for epsilon for the derivation? (current value : {}) '.format(_EP2))or _EP2)
                        break
                    except ValueError:
                        print("Oops!  That was no valid number.  Try again...")

                dYdX = TVRegDiff(data=_y ,itern=ITERN, 
                    alph=_ALPH, dx=dX, 
                    ep=_EP,
                    scale=SCALE ,
                    plotflag=False, 
                    precondflag=PRECOND,
                    diffkernel=KERNEL,
                    u0=np.append([0],np.diff(_y)),
                    cgtol = 1e-4)    

    
                #Conversion de la pente (en g/jour) en mmol/s
                k= (dYdX/18.01528)*(1000/60) #ici c'est en minutes (60*60*24)
                #Calcul VPD en kpa (Patm = 101.325 kPa)
                VPD =0.1*((6.13753*np.exp((17.966*df[T].values/(df[T].values+247.15)))) - (df[RH].values/100*(6.13753*np.exp((17.966*df[T].values/(df[T].values+247.15)))))) 
                #calcul gmin mmol.s
                gmin_ = -k * df[PATM].values/VPD
                #calcul gmin en mmol.m-2.s-1
                gmin = gmin_ / df[AREA].values
                #df['raw_slope'] = dYdX
                #df['gmin'] = gmin                     
                dGmin = TVRegDiff(data=gmin ,itern=ITERN, 
                                alph=_ALPH2, dx=dX, 
                                ep=_EP2,
                                scale=SCALE ,
                                plotflag=False, 
                                precondflag=PRECOND,
                                diffkernel=KERNEL,
                                u0=np.append([0],np.diff(gmin)),
                                cgtol = 1e-4)
                ddGmin = dGmin
                peaks, _ = find_peaks(ddGmin, distance=DIST, prominence = np.max(ddGmin)/PROM)
                peaks2, _ = find_peaks(-ddGmin, distance=DIST, prominence = np.max(ddGmin)/PROM)
                peaks = np.concatenate((peaks, peaks2), axis=None)

                self._plot_tvregdiff(_X=_X[:], _y=gmin[:], _y2 = ddGmin, peaks=peaks)   
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
        df['d_gmin'] =  ddGmin

        #print(df)
        if len(peaks)>0:
            try:
                df['Peaks'] = np.array_str(_X[peaks])
            except:                
                df['Peaks'] = ' '.join(map(str,peaks))
        else:
            df['Peaks'] = 'NoPeak'
 
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
        self.df_save = pd.DataFrame(columns = range(0,19))
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
        
        if not os.path.exists('output_files'):
                os.makedirs('output_files')
        self.df_save.to_csv('output_files/gmin.csv')
                    #dfe = 
                    # df_save.columns = dfe.columns
                    # df_save = pd.concat([df_save, dfe], axis = 0, ignore_index = True)
                    # df_save.to_csv('gmin.csv')

  


            