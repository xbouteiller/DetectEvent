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
print('------------------------------------------------------------------------')
print('---------------                                    ---------------------')
print('---------------            LeafConductance         ---------------------')
print('---------------                  V0.1              ---------------------')
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
LAG_DIV = WIND_DIV * 45
BOUND = 'NotSet'
FRAC_P = 0.1
DELTA_MULTI = 0.01
PAUSE_GRAPH = 8

ITERN=20 
ALPH=10000
EP=1e-9


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
        List of actions

        1. Detect changes in curve (RMSE approach)
        2. Compute conductance (robust differential approach)
        3. Exit
        """)

    def _quit(self):
        print("Thank you for using your LeafConductance today.")
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
        ax1.plot(self.Xselected, self.yselected, color=color, linestyle='-', marker='.', label = 'Weight (g)')
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
    
    def _change_det(self, df, COL_Y='standard'):

        if df.shape[0] < 100:
            _wind = int(df.shape[0]/6)
            _lag = 1# int(df.shape[0]/4)
        else:
            _wind = int(df.shape[0]/WIND_DIV)
            _lag = int(df.shape[0]/LAG_DIV)
        _X = df['delta_time'].copy().values
        if COL_Y == 'standard':
            _y = df[YVAR].copy().values
        else:
             _y = df[COL_Y].copy().values


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
                    self._parse_samples(dffile = dffile, FUNC = self._change_det)
                    pd.DataFrame(self.global_score, columns = ['Sample_ID', 'Wind_of_Change']).to_csv('global_score.csv')


        
        
    def _plot_tvregdiff(self, _X, _y, _y2, peaks, ax2_Y =r'$Gmin (mmol.m^{-2}.s^{-1})$', ax2_label = 'Gmin' ):
        
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
        # plt.pause(PAUSE_GRAPH)
        plt.waitforbuttonpress(0)
        # input()
        plt.close()   
    
    
    def _tvregdiff(self,df):

        _X = df['delta_time'].copy().values
        _y = df[YVAR].copy().values

        dX = _X[1] - _X[0]


        if len(_X)<200:   # MAYBE HYPERPARAMETERS CAN BE DEFINED OTHERLY
            DIV_ALPH = 100000 # 1000
            DIV_ALPH2 = 100 # 1000
            DIST = 10
            PROM = 20 #10
        else:
            DIV_ALPH = 1 #10
            DIV_ALPH2= 100
            DIST = 100 #200
            PROM = 4#4
        

        dYdX = TVRegDiff(data=_y ,itern=ITERN, 
                        alph=ALPH/DIV_ALPH, dx=dX, 
                        ep=EP,
                        scale='small' ,
                        plotflag=False, 
                        precondflag=False,
                        diffkernel='abs',
                        cgtol = 1e-5)
        
        #Conversion de la pente (en g/jour) en mmol/s
        k= (dYdX/18.01528)*(1000/60) #ici c'est en minutes (60*60*24)

        #Calcul VPD en kpa (Patm = 101.325 kPa)
        VPD =0.1*((6.13753*np.exp((17.966*df[T].values/(df[T].values+247.15)))) - (df[RH].values/100*(6.13753*np.exp((17.966*df[T].values/(df[T].values+247.15)))))) 

        #calcul gmin mmol.s
        gmin_ = -k * df[PATM].values/VPD

        #calcul gmin en mmol.m-2.s-1
        gmin = gmin_ / df[AREA].values
        df['raw_slope'] = dYdX
        #df['gmin'] = gmin
        
      
        dGmin = TVRegDiff(data=gmin ,itern=ITERN, 
                        alph=ALPH/DIV_ALPH2, dx=dX, 
                        ep=EP,
                        scale='small' ,
                        plotflag=False, 
                        precondflag=False,
                        diffkernel='abs',
                        cgtol = 1e-5)
        # do we really need a second order derivative ?
        # ddGmin = TVRegDiff(data=dGmin,itern=ITERN, 
        #                 alph=ALPH/DIV_ALPH2, dx=dX, 
        #                 ep=EP,
        #                 scale='small' ,
        #                 plotflag=False, 
        #                 precondflag=False,
        #                 diffkernel='abs',
        #                 cgtol = 1e-5)


        ddGmin = dGmin
        #ddGmin = np.diff(gmin,1)
        peaks, _ = find_peaks(ddGmin, distance=DIST, prominence = np.max(ddGmin)/PROM)
        peaks2, _ = find_peaks(-ddGmin, distance=DIST, prominence = np.max(ddGmin)/PROM)
        peaks = np.concatenate((peaks, peaks2), axis=None)
        #print('peaks ',peaks)

        df['d_gmin'] = ddGmin
        #print(df)
        if len(peaks)>0:
            df['Peaks'] = np.array_str(_X[peaks])
        else:
            df['Peaks'] = 'NoPeak'
        #peaks = peaks + 
        self._plot_tvregdiff(_X=_X, _y=gmin, _y2 = ddGmin, peaks=peaks)   
        
        self.df_save.columns = df.columns
        self.df_save = pd.concat([self.df_save, df], axis = 0, ignore_index = True)
        

        #self._plot_tvregdiff(_X=_X, _y=ddGmin, ax2_Y =r'$Gmin (mmol.m^{-2}.s^{-3})$', ax2_label = 'double diff Gmin') 
        
        # self._change_det(df, COL_Y='gmin')

        return df

    def robust_differential(self):    
        dimfolder = len(self.listOfFiles)
        li_all = []
        self.df_save = pd.DataFrame(columns = range(0,15))
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
        
        self.df_save.to_csv('gmin.csv')
                    #dfe = 
                    # df_save.columns = dfe.columns
                    # df_save = pd.concat([df_save, dfe], axis = 0, ignore_index = True)
                    # df_save.to_csv('gmin.csv')

  


            