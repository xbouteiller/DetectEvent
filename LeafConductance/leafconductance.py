import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from loess.loess_1d import loess_1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from tvregdiff import TVRegDiff
import pandas as pd
from scipy.optimize import curve_fit
import time

print('------------------------------------------------------------------------')
print('---------------                                    ---------------------')
print('---------------            LeafConductance         ---------------------')
print('---------------                  V0.1              ---------------------')
print('---------------                                    ---------------------')
print('------------------------------------------------------------------------')
time.sleep(2)

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
        "1": self.change_detection,
        "2": self.compute_conductance,
        "3": self.extract_strings,
        "4": self.erase,
        "5": self.extract_strings_and_nums
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
                f.endswith('.csv') and (re.search(r'CONDUCTANCE|conductance', pd.read_csv(os.path.join(d, f),sep=SEP,nrows=0).columns[0]) )]

    
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
                    # print(file_root)
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
                    f.endswith('.csv') and (re.search(r'CONDUCTANCE|conductance', pd.read_csv(os.path.join(d, f),sep=SEP,nrows=0).columns[0]) )]
                    self.listOfFiles.append(file_root)
                    # print(file_root)
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


    def _parse_samples(self, FUNC): 
        import pandas as  pd
        import numpy as np       

        for sample in self.file[SAMPLE_ID].unique():
            self.sample = sample
            df = self.file[self.file[SAMPLE_ID]]
            df[TIME_COL] = pd.to_datetime(df[TIME_COL] , infer_datetime_format=True)  
            df['delta_time'] = (df[TIME_COL]-df[TIME_COL].shift()).fillna(0)            
            FUNC(df)



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
        what_to_do = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2')
    
        if what_to_do=='1':
            self.global_score.append([self.sample,'Discarded', 'Discarded'])
        if what_to_do=='2':
            self.global_score.append([self.sample,Xidx_int[0], Xidx_int[1]])



        return idx, Xidx, Xidx_int 

    def _change_det(self, df):
        _wind = int(df.shape[0]/WIND_DIV)
        _lag = int(df.shape[0]/LAG_DIV)
        self._X = df['delta_time']
        self._y = df[XVAR]

        score_l, mean_start_l = self._sliding_window_pred(_X, _y, window=_wind, lag=_lag, mode = 'linear')
        score_e, mean_start_e = self._sliding_window_pred(_X, _y, window=_wind, lag=_lag, mode = 'exp')
        idx, Xidx, Xidx_int = self._detect_crossing_int(score_l, score_e, mean_start_l)

    def change_detection(self):
        print('change_detection\n')
        self._parse_samples(self, FUNC = self._change_det)



  


            


    def compute_conductance(self):
        import numpy as np
        import pandas as pd
        print('choose to do modify')

        while True:
            try:
                nval = float(input('how many values do you want to modify ? '))
                break
            except ValueError:
                print("Oops!  That was no valid number.  Try again...")

        for i in np.arange(0,nval):

            tobemodified = input('Which value do you want to change ? ')
            while True:
                if tobemodified in self.frame[self.i].unique().tolist():
                    break
                else:
                    print("Oops! identifiant not existing choose one among : {}".format(self.frame[self.i].unique().tolist()))
                    tobemodified = input('Which value do you want to change ? ')

            newvalue = input('What is the new value ? ')
            self.frame.loc[self.frame[self.i]==tobemodified,self.i] = newvalue
            print('new values are {}'.format(self.frame[self.i].unique()))
            input('press any key to continue')

    def extract_strings(self):
        print('choose to extract strings')
        import re
        import numpy as np

        print('col',self.frame[self.i])


        self.frame[self.i]=self.frame[self.i].str.extract('([a-zA-Z]+)', expand = False)

        print('modified to {}'.format(self.frame[self.i].unique()))
        inp=input('press any key to continue --- or enter 1 to modify values ---')
        if inp == str(1):
            self.modify()
        else:
            print('no values to be modified')

    def extract_strings_and_nums(self):
        print('choose to extract strings and numbers')
        import re
        import numpy as np
        # print('col',self.frame[self.i])
        reg = self.frame[self.i].str.extract('([a-zA-Z]+)\W(\d+)', expand = False)

        self.frame[self.i]=reg[0]
        self.frame['Sample_ref_2']=reg[1]#.astype('int')
        print('modified {} to {}'.format(self.i, self.frame[self.i].unique()))
        print('modified "Sample_ref_2" to {}'.format(self.frame['Sample_ref_2'].unique()))

        inp=input('press any key to continue --- or enter 1 to modify {} values ---'.format(self.i))
        if inp == str(1):
            self.modify()
        else:
            print('no values to be modified')

    def erase(self):
        import numpy as np
        import pandas as pd
        print('choose to do erase rows')
        while True:
            try:
                nval = float(input('how many values do you want to erase ? '))
                break
            except ValueError:
                print("Oops!  That was no valid number.  Try again...")


        for i in np.arange(0,nval):
            tobemodified = input('Which row value do you want to erase ? ')
            while True:
                if tobemodified in self.frame[self.i].unique().tolist():
                    break
                else:
                    print("Oops! values not existing in {} choose one among : {}".format(self.i, self.frame[self.i].unique().tolist()))
                    tobemodified = input('Which row value do you want to erase ? ')

            self.frame=self.frame[self.frame[self.i]!=tobemodified]
            print('new values are {}'.format(self.frame[self.i].unique()))
            input('press any key to continue')

    def _clean_plc(self):
        import pandas as pd
        # print(self.frame['PLC'].str.isnumeric().values)
        # print(self.frame.shape)
        # self.frame=self.frame[self.frame['PLC'].str.isnumeric().values]
        # self.frame['PLC']=pd.to_numeric(self.frame['PLC'].values)
        # print(self.frame.shape)
        # self.frame['PLC']<=100
        # print(self.frame.shape)
        # print(self.frame['PLC'])
        # print(self.frame['PLC'].str.isdigit().values)
        # print(pd.to_numeric(self.frame['PLC'], errors='coerce').notnull())
        # print(pd.to_numeric(self.frame['PLC'], errors='coerce').notnull().shape)
        try:
            n0r=self.frame.shape[0]
            self.frame=self.frame[pd.to_numeric(self.frame['PLC'], errors='coerce').notnull()]
            self.frame['PLC']=pd.to_numeric(self.frame['PLC'].values)
            n1r=self.frame.shape[0]
            print('\n\nremoving non numeric PLC, {} rows removed'.format(n1r-n0r))
            self.frame=self.frame[self.frame['PLC']<=100]
            n2r=self.frame.shape[0]
            print('removing PLC values higher than 100, {} rows removed'.format(n2r-n1r))
        except:
            print('an exception occured when trying to clean PLC values')

    def clean_plc(self):
            import pandas as pd
            print('\n ---------------------------------------------------------------------')
            print('Cleaning PLC values')
            self._clean_plc()

    def check_frame_num(self):
        import pandas as pd
        all_cn, cn = self._check_num(_df = self.frame , _col= num_col)
        # cn = self._check_num(_df = self.frame , _col= num_col)[1]

        # if not all_cn:
        #     print('\n -----------------------')
        #     [print('col {} is Numeric'.format(i)) if j else print('col {} is not Numeric'.format(i)) for i, j in zip(num_col, cn)]
        #     input('press any key to continue')
        if not all_cn:
            print('\n\n ---------------------------------------------------------------------')
            print('checking numerical columns\n')
            [print('col {} is Numeric'.format(i)) if j else print('col {} is not Numeric'.format(i)) for i, j in zip(num_col, cn)]

            for i, j in zip(num_col, cn):
                if not j:
                    self.i=i
                    # I want to register on my log the message recived on ORIGINAL VALUE
                    # print(self.frame[self.i].head())
                    mask = pd.to_numeric(self.frame[self.i], errors='coerce').isna()
                    #if possible missing values
                    # mask = pd.to_numeric(df['ORIGINAL_VALUE'].fillna('0'), errors='coerce').isna()
                    L = self.frame.loc[mask, self.i].tolist()
                    print('\nin col {}'.format(self.i))
                    print ("Not converted to numeric values are: " + ", ".join(L))

                    print('What do you want to do ? ')
                    self.run()



    def check_frame_group(self):
            all_cg, cg= self._check_group(_df = self.frame , _col = group_col)#[0]
            # cg = self._check_group(_df = self.frame , _col = group_col)[1]
            if not all_cg:
                print('\n\n ---------------------------------------------------------------------')
                print('checking categorical columns\n')
                [print('label of col {} is unique'.format(i)) if j else print('label of col {} is NOT unique'.format(i)) for i, j in zip(group_col, cg)]

                for i, j in zip(group_col, cg):
                    if not j:
                        self.i=i
                        print('labels of col {} are {}\nWhat do you want to do ?'.format(self.i, self.frame[self.i].unique()))
                        self.run()

    def _compute_empty(self):
        self.frame[empty_col]=self.frame[empty_col].fillna(self.frame['Sample_ref_1']-self.frame['Sample_ref_1'].min())
        print('new value in {} are {}'.format(empty_col, self.frame[empty_col].unique()))
        input('press any key to continue')

    def _manual_empty(self):
        for man in self.frame['Sample_ref_1'].unique():
            if any(self.frame.loc[self.frame['Sample_ref_1']==man, empty_col].isna()):
                while True:
                    try:
                        newvalue= float(input('What is the identifiant for "ref number 2" for individual {}: '.format(man)))
                        break
                    except ValueError:
                        print("Oops!  That was no valid number.  Try again...")
                self.frame.loc[self.frame['Sample_ref_1']==man,empty_col]=newvalue
            else:
                pass

        print('new value in {} are {}'.format(empty_col, self.frame[empty_col].unique()))
        input('press any key to continue')

    def _extract_empty(self):
        self.frame[empty_col]=self.frame[empty_col].fillna(self.frame['Comment'].str.extract('(\d+)', expand = False))
        print('new value in {} are {}'.format(empty_col, self.frame[empty_col].unique()))
        input('press any key to continue')

    def check_frame_empty(self):
        any_empty, empty = self._check_empty(_df = self.frame , _col= empty_col)
        ae=0
        while any_empty:
            print('\n ---------------------------------------------------------------------')
            print('parsing list of files from : {}\n'.format(self.presentfile))

            print('{} contains empty values'.format(empty_col))
            print('value in Comment columns are {}'.format(self.frame['Comment'].unique()))
            print('''
            --------------------
            List of actions

            1: do nothing
            2: calculate from 1 to n
            3: extract numbers from Comment columns
            4: enter values manually
            ''')
            wtd = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2', '3', '4'))
            if wtd == '1':
                break
            if wtd == '2':
                self._compute_empty()
                any_empty = self._check_empty(_df = self.frame , _col= empty_col)[0]
            if wtd == '3':
                self._extract_empty()
                any_empty = self._check_empty(_df = self.frame , _col= empty_col)[0]
            if wtd == '4':
                self._manual_empty()
                any_empty = self._check_empty(_df = self.frame , _col= empty_col)[0]
            ae+=1
        if ae>0:
            print('\nExiting from empty verification\nnew value in {} are {}'.format(empty_col, self.frame[empty_col].unique()))


    def _inactive_indiv(self):
        while True:
            try:
                newvalue= float(input('What is the identifiant of the individual that you want to inactive ? '))
                break
            except ValueError:
                print("Oops!  That was no valid number.  Try again...")

        while True:
            if newvalue in self.frame['Sample_ref_1'].unique():
                break
            else:
                print("Oops! identifiant not existing choose one among : {}".format(self.frame['Sample_ref_1'].unique().tolist()))
                while True:
                    try:
                        newvalue= float(input('What is the identifiant of the individual that you want to inactive ? '))
                        break
                    except ValueError:
                        print("Oops!  That was no valid number.  Try again...")

        self.frame.loc[self.frame['Sample_ref_1']==newvalue,'Note']='yes'


    def inactive_indiv(self):
        print('\n ---------------------------------------------------------------------')
        print('\nDo you want to inactive (turn to yes) some individuals ? ')

        print('''
        --------------------
        List of actions

        1: no, escape and continue
        2: yes, inactive some individuals
        ''')
        wtd = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2'))

        if wtd == '2':
            inactive = True
        else:
            inactive = False

        while inactive:
            self._inactive_indiv()
            exit = input('press any key to continue or enter -- exit -- to stop ')
            if exit == 'exit' or exit == 'e':
                inactive = False

        L = self.frame.loc[self.frame['Note']=='yes','Sample_ref_1'].unique().tolist()
        L = [str(i) for i in L]
        print('\nExiting from inactivating individuals\ninactivated individuals are : ' + ", ".join(L))


    def _change_values(self):

        # num_col = ['PLC','Meas_cavispeed_rpm','Pressure_Mpa']
        # group_col=['Sampling_location', 'Treatment', 'Operator']
        # empty_col='Sample_ref_2'


        print('''
        --------------------
        Columns that you can change

        1: Sample_ref_2 (tree number)
        2: Sampling_location
        3: Treatment
        4: Operator
        5: Note
        6: exit
        ''')
        ctc = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2','3','4','5','6'))

        if ctc == '1':
            col_to_change = 'Sample_ref_2'
        if ctc == '2':
            col_to_change = 'Sampling_location'
        if ctc == '3':
            col_to_change = 'Treatment'
        if ctc == '4':
            col_to_change = 'Operator'
        if ctc == '5':
            col_to_change = 'Note'
        if ctc == '6':
            col_to_change = '-- exiting--'

        print('\nmodified column will be : {}'.format(col_to_change))

        if ctc == '6':
            pass
        else:
            while True:
                try:
                    newvalue= float(input('What is the identifiant of the individual that you want to change ? '))
                    break
                except ValueError:
                    print("Oops!  That was no valid number.  Try again...")

            while True:
                if newvalue in self.frame['Sample_ref_1'].unique():
                    break
                else:
                    print("Oops! identifiant not existing choose one among : {}".format(self.frame['Sample_ref_1'].unique().tolist()))
                    while True:
                        try:
                            newvalue= float(input('What is the identifiant of the individual that you want to change ? '))
                            break
                        except ValueError:
                            print("Oops!  That was no valid number.  Try again...")

            modified_value= input('\nWhat is the new value that you want to change for individual {} in column {} ? '.format(newvalue, col_to_change))
            self.frame.loc[self.frame['Sample_ref_1']==newvalue,col_to_change]=modified_value
            # print(self.frame.loc[self.frame['Sample_ref_1']==newvalue,col_to_change])
            assert self.frame.loc[self.frame['Sample_ref_1']==newvalue,col_to_change].unique().tolist() == [modified_value], 'pb with new value'

    def manual_change(self):
                print('\n ---------------------------------------------------------------------')
                print('\nDo you want to change manually some values ? ')

                print('''
                --------------------
                List of actions

                1: no, escape and continue
                2: yes, change manually some individuals values
                ''')
                wtd = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2'))

                if wtd == '2':
                    change = True
                else:
                    change = False

                while change:
                    self._change_values()
                    exit = input('press any key to continue or enter -- exit -- to stop ')
                    if exit == 'exit' or exit == 'e':
                        change = False


    def check_unicity(self):
        print('\n ---------------------------------------------------------------------')
        print('\nChecking unicity')

        print('REP column already exists ? {}'.format('REP' in self.frame.columns))
        if 'REP' in self.frame.columns:
            self.frame['REP']=self.frame['REP'].fillna(1)
            print('empty REP column values filled with 1')
        else:
            self.frame['REP']=1
            print('REP column created')

        pb = []
        for camp in self.frame['Campaign_name'].unique():
            for loc in self.frame['Sampling_location'].unique():
                for sp in self.frame['Species'].unique():
                    for tr in self.frame['Treatment'].unique():
                        for cav in self.frame['Sample_ref_2'].unique():
                            for rep in self.frame['REP'].unique():
                                nval = len(self.frame.loc[(self.frame['Campaign_name']==camp) & (self.frame['Sampling_location']==loc) & (self.frame['Species']==sp) & (self.frame['Treatment']==tr) & (self.frame['Sample_ref_2']==cav) & (self.frame['REP']==rep),'Sample_ref_1'].unique().tolist())
                                # print(nval)
                                if nval >1:
                                    pb.append([camp,loc,sp,tr,rep,cav,nval])


        # print(self.frame['REP'])

        for n in pb:
            cavit_number = self.frame.loc[(self.frame['Campaign_name']==n[0]) & (self.frame['Sampling_location']==n[1]) & (self.frame['Species']==n[2]) & (self.frame['Treatment']==n[3]) & (self.frame['REP']==n[4]) & (self.frame['Sample_ref_2']==n[5]),'Sample_ref_1'].unique().tolist()
            tree_number = self.frame.loc[(self.frame['Campaign_name']==n[0]) & (self.frame['Sampling_location']==n[1]) & (self.frame['Species']==n[2]) & (self.frame['Treatment']==n[3]) & (self.frame['REP']==n[4]) & (self.frame['Sample_ref_2']==n[5]),'Sample_ref_2'].unique().tolist()
            print('''
                 ------------------
                 description

                 campaign: {}
                 site: {}
                 species: {}
                 treament: {}
                 repetition: {}
                 sample ref 1 (cavit number) : {}
                 sample ref 2 (tree number): {}
                  '''.format(n[0],n[1],n[2],n[3],n[4],cavit_number, tree_number))

            print('''
                --------------------
                Error sample ref 1 don't refer to an unique tree
                What do you want to do ?

                1: nothing, escape and continue
                2: change manually some individual values (sample ref 2)
                3: automatically compute repetition numbers for each sample ref 1
                4: manual change (if you want to change other column value e.g treatment)
                ''')
            wtd = self._get_valid_input('What do you want to do ? Choose one of : ', ('1','2', '3', '4'))

            if wtd == '6':
                pass

            if wtd == '2':
                for caval in cavit_number:
                    while True:
                        try:
                            newvalue= float(input('What is the new identifiant for "Sample_ref_2" for individual {}: '.format(caval)))
                            break
                        except ValueError:
                            print("Oops!  That was no valid number.  Try again...")
                    self.frame.loc[self.frame['Sample_ref_1']==caval,'Sample_ref_2']=newvalue

                print('new value in {} are {}'.format('Sample_ref_2', self.frame['Sample_ref_2'].unique()))
                input('press any key to continue')

            if wtd == '3':
                repet = 1
                for ca in cavit_number:
                    # print(self.frame.loc['sample ref 1'==ca,['REP']])
                    self.frame.loc[self.frame['Sample_ref_1']==ca,['REP']]=repet
                    repet += 1

                print('rep values for {} are {}'.format(cavit_number, self.frame.loc[[True if ca in cavit_number else False  for ca in self.frame.Sample_ref_1],'REP'].unique()))
                input('press any key to continue')

            if wtd == '4':
                self.manual_change()

    def append_values(self):
        '''
        method for filling the lists
        '''

        import numpy as np
        import pandas as pd


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

                    # print(df)
                    li.append(df)

                self.frame = pd.concat(li, axis=0, ignore_index=True, sort=False)
                print('shape of frame is {}'.format(self.frame.shape))
                self.check_frame_num()
                self.check_frame_group()
                self.check_frame_empty()
                self.inactive_indiv()
                self.manual_change()
                self.clean_plc()
                self.check_unicity()
                li_all.append(self.frame)
                #check integrity
            else:
                print('skip empty folder')
                pass

        self.final_frame = pd.concat(li_all, axis=0, ignore_index=True, sort=False)
        print('shape of final frame is {}'.format(self.final_frame.shape))

        return self.final_frame

    def _summarize_df(self):
        import pandas as pd
        import numpy as np

        print('\n\n---------------------------------------------------------------------')
        print('data frame summary')
        print('\nnumerical columns\n')
        for i in num_col:
            print('- for : {0}, mean is : {1:.2f}, sd is : {2:.2f}, min is : {3:.2f}, max is : {4:.2f}'.format(i,np.mean(self.frame[i]),np.std(self.frame[i]),np.min(self.frame[i]),np.max(self.frame[i])))

        print('\ncategorical columns')
        for i in group_col+['Note']:
            print('\n- for : {}, categories are : {}'.format(i,self.frame[i].unique().tolist()))
            print(self.frame[i].value_counts())

    def save_finaldf(self):
        '''
        saved the concatened df into a csv file
        '''
        import pandas as pd
        import os
        from tkinter import Tk
        from tkinter.filedialog import asksaveasfilename
        # import time

        self._summarize_df()
        print('''
        --------------------
        Last step
        Do you want to save the data frame or to restart the process ?

        1. save
        2. restart verifs
        ''')

        finchoi=self._get_valid_input('What is your choice ? Choose one of : ', ('1','2'))

        if finchoi == '1':
            pass
        else:
            while finchoi == '2':
                self.frame = self.final_frame
                self.check_frame_num()
                self.check_frame_group()
                self.check_frame_empty()
                self.inactive_indiv()
                self.manual_change()
                self.clean_plc()
                self.check_unicity()
                self.final_frame = self.frame

                self._summarize_df()
                print('''
                --------------------
                Last step
                Do you want to save the data frame or to restart the process ?

                1. save
                2. restart
                ''')

                finchoi=self._get_valid_input('What is your choice ? Choose one of : ', ('1','2'))

        print('\n\n---------------------------------------------------------------------')
        if self.file_or_folder== '1':
            print('initialdir {} '.format(self.path))
            print('initialfile {} '.format(os.path.basename(os.path.normpath(self.path))))
            idir=self.path
            ifile=os.path.basename(os.path.normpath(self.path))

        if self.file_or_folder== '2' or self.file_or_folder== '3':
            print('initialdir {} '.format(os.path.dirname(self.path)))
            print('initialfile {} '.format(os.path.basename(self.path)))
            idir=os.path.dirname(self.path)
            ifile=os.path.basename(self.path)

        Tk().withdraw()
        FileSaveName=asksaveasfilename(defaultextension='.csv', filetypes=[("csv files", '*.csv')],
                initialdir=idir,
                initialfile=ifile,
                title="Choose filename")

        try:
            print('\nfile name is : {} '.format(os.path.basename(FileSaveName)))
            print('file is saved to : {} '.format(os.path.dirname(FileSaveName)))
        except:
            print('no file name detected, saved to DefaultTable.csv')
            FileSaveName = input('enter final file name : ') or 'DefaultTable'
            FileSaveName += '.csv'

        self.final_frame.to_csv(FileSaveName,index=False, header=True) #which sep ? , sep=';'
        print('saved file {}\n'.format(FileSaveName))
