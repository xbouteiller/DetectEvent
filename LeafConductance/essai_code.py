import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from loess.loess_1d import loess_1d
from statsmodels.nonparametric.smoothers_lowess import lowess
# from tvregdiff import TVRegDiff
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
WIND_DIV = 5
LAG_DIV = WIND_DIV * 10
BOUND = ([0.0,  0.0], [0.,  0.0])



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

        # #drop full na
        # self.file = self.file.dropna(axis = 0, how = 'all')

        # # convert to numeric if possible
        # self.file = self.file.apply(lambda x: pd.to_numeric(x, errors ="ignore"))

        # # lower strings
        # self.file = self.file.applymap(lambda s:s.lower() if (isinstance(s, str) and s!='None')  else s)

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

        Type 1 to start and parse a folder

        1: Parse a folder
        ''')
        self.file_or_folder = self._get_valid_input('Type 1 to start : ', ('1'))

        if self.file_or_folder== '1':
            Tk().withdraw()
            folder = askdirectory(title='What is the root folder that you want to parse ?')
            self.path = folder
            print('\n\n\nroot path is {}'.format(self.path))

            print('''
            which method do you want to use for detecting cavisoft files ?

            1: Detect all CSV files 
            2: Detect string 'CONDUCTANCE' string in the first row
            ''')
            self.method_choice = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2'))

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


 



    def _RMSE(self, Ypred, Yreal):
        rmse = np.sqrt(np.sum(np.square(Ypred-Yreal))/np.shape(Ypred)[0])
        return rmse
    def _fit_and_pred(self,X, y):
        from sklearn.linear_model import LinearRegression
        Xarr = np.array(X).reshape(-1,1)
        yarr = np.array(y).reshape(-1,1)
        reg = LinearRegression().fit(Xarr, yarr)
        pred = reg.predict(Xarr)
        rmse = self._RMSE(pred, yarr)
        return rmse
    
    def _sliding_window_pred(self, X, y, window, lag, mode, b=BOUND):
        Xmax = np.shape(X)[0]-window+1
        start = np.arange(0, Xmax, lag)

        if mode == 'linear':
            mean_start = X[[int(s + window/2) for s in start]]
            score = [self._fit_and_pred(X[s:s+window], y[s:s+window]) 
                    for s in start]    
            return score, mean_start
        
        if mode == 'exp':
            mean_start = X[[int(s + window/2) for s in start]]
            
            if BOUND == ([0.0,  0.0], [0.,  0.0]):
                reg = curve_fit(self._func, X[lag:lag+window], y[lag:lag+window])[0]
                A, B = reg 
                b = ([A-0.1*A,B/10],[A+0.1*A, B*10])
            else:
                b=BOUND

            score = [fit_and_pred_exp(X[s:s+window], y[s:s+window], b=b, p = p) 
                    for s in start]    
        
        return score, mean_start


    def _func(self, x, a, b):
        return a * np.exp(-b * x) 
    
    def _detect_b(self, X, y):
        Xarr = np.array(X).reshape(-1)
        yarr = np.array(y).reshape(-1)
        reg = curve_fit(func, Xarr, yarr)[0]



    def _fit_and_pred_exp(self, X, y):
        Xarr = np.array(X).reshape(-1)
        yarr = np.array(y).reshape(-1)
        b=self._detect_b(Xarr,yarr)
        reg = curve_fit(self._func, Xarr, yarr, bounds=b)[0]
        A, B = reg     
        pred = A * np.exp(-B * Xarr)
        rmse = RMSE(pred, yarr)
        return rmse

    def _dcross(self, Yl, Ye):
        idx = np.argwhere(np.diff(np.sign(Yl - Ye))).flatten() 
        return idx

    def _detect_crossing_int(self, Yexp, Ylin, X):
        Ylin=np.array(Ylin)
        Yexp=np.array(Yexp)
        X=np.array(X)  
        
        idx = self._dcross(Ylin, Yexp)
        Xidx=X[idx]    
        idx_int = [[i, i+1] for i in idx]
        Xidx_int = [[X[i], X[i+1] ]for i in idx]
        Yidx=Ylin[idx]

        
        plt.plot(X, Ylin)
        plt.plot(X, Yexp)
        
        for i in np.arange(0,len(idx)):
            plt.hlines(Yidx[i], Xidx_int[i][0], Xidx_int[i][1], lw=4, color = 'red')
        plt.show()     
        
        print('\nInterval method')
        for i in np.arange(0,len(idx)):
            print('detected changes between times : {} - {}'.format(Xidx_int[i][0], Xidx_int[i][1]))
        
        print('''
            Do you want to keep the crossing value ?

            1: No, discard 
            2: Yes, save
            3. Select value manually on graph (WORK IN PROGRESS)
            ''')
        what_to_do = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2'))
    
        if what_to_do=='1':
            self.global_score.append([self.sample,'Discarded', 'Discarded'])
        if what_to_do=='2':
            self.global_score.append([self.sample,Xidx_int[0], Xidx_int[1]])

        return idx, Xidx, Xidx_int 

    def _change_det(self, df):

        if df.shape[0] < 100:
            _wind = int(df.shape[0]/2)
            _lag = int(df.shape[0]/4)
        else:
            _wind = int(df.shape[0]/WIND_DIV)
            _lag = int(df.shape[0]/LAG_DIV)
        _X = df['delta_time']
        _y = df[YVAR]

        score_l, mean_start_l = self._sliding_window_pred(_X, _y, window=_wind, lag=_lag, mode = 'linear')
        score_e, mean_start_e = self._sliding_window_pred(_X, _y, window=_wind, lag=_lag, mode = 'exp')
        idx, Xidx, Xidx_int = self._detect_crossing_int(score_l, score_e, mean_start_l)
   
    def _parse_samples(self, dffile, FUNC): 
        import pandas as  pd
        import numpy as np       

        for sample in dffile[SAMPLE_ID].unique():
            self.sample = sample
            df = dffile.loc[dffile[SAMPLE_ID]==sample,:].copy()
            df['TIME_COL2'] = pd.to_datetime(df[TIME_COL] , infer_datetime_format=True)  
            df['delta_time'] = (df['TIME_COL2']-df['TIME_COL2'].shift()).fillna(0.0)   
            df['delta_time'] = df['delta_time'].apply(np.float32)         
            FUNC(df)

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
                    # print(elem)
                    if self.file_or_folder== '1':
                        skip=1
                    else:
                        skip=0
                    try:
                        df = ParseFile(path = elem, skipr=skip).clean_file()
                    except:
                        encodi='latin'
                        df = ParseFile(path = elem, skipr=skip, encod=encodi).clean_file()

                    if df.shape[1] == 1:
                        separ=';'
                        try:
                            df = ParseFile(path = elem, sepa=separ, skipr=skip).clean_file()
                        except:
                            encodi='latin'
                            df = ParseFile(path = elem, skipr=skip, sepa=separ, encod=encodi).clean_file()


                    self._parse_samples(dffile = df, FUNC = self._change_det)



  


            