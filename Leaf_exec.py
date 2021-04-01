#!/home/xavier/anaconda3/bin/python
# python Leaf_exec.py -tr 4 -fr 2 -td 2 -fd 2 
# python Leaf_exec.py --transfo_rmse 4 --fit_rmse 2 --transfo_diff 2 --fit_diff 2 


def float_range(mini,maxi):
    """Return function handle of an argument type function for 
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - maximum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker


if __name__=="__main__":


    #%% Dependencies
    import os
    import argparse
    from LeafConductance.leafconductance import ParseFile, ParseTreeFolder


    # Parser for defining parameters values using the shell
    # default  values can be modified directly HERE in the script or using the shell command line prompt
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-tc','--time_col', default='date_time', 
                        help='which time col', 
                        type = str)

    parser.add_argument('-sid','--sample_id', default='sample_ID', 
                        help='which sample id col', 
                        type = str)

    parser.add_argument('-y','--yvar', default='weight_g', 
                        help='which Y col', 
                        type = str)
                        
    parser.add_argument('-t','--temp', default='T_C', 
                        help='which temp col', 
                        type = str)

    parser.add_argument('-rh','--rh', default='RH', 
                        help='which RH col', 
                        type = str)
                        
    parser.add_argument('-p','--patm', default='Patm', 
                        help='which P col', 
                        type = str)

    parser.add_argument('-a','--area', default='Area_m2', 
                        help='which area col', 
                        type = str)

    parser.add_argument('-it','--iter_n', default=10000, 
                        help='max number of iteration"', 
                        type = int)

    parser.add_argument('-al','--alpha', default=1e8, 
                        help='value of alpha for derivating the signal', 
                        type = float)
    
    parser.add_argument('-ep','--epsilon', default=1e-6, 
                        help='value of epsilon', 
                        type = float)

    parser.add_argument('-dm','--diff_method', default='sq', 
                        help='which method to use for differentiating', 
                        choices = ["abs", "sq"],
                        type = str)

    parser.add_argument('-tr','--transfo_rmse', default='1', 
                        help='which method to use for transformating data\n1. None\n2.log lin\n3. exp part exp\n.4. log lin & exp exp', 
                        choices = ["1", "2", '3', '4'],
                        type = str)

    parser.add_argument('-td','--transfo_diff', default='1', 
                        help='which method to use for transformating data\n1. None\n2.log lin\n3. exp part exp\n.4. log lin & exp exp', 
                        choices = ["1", "2", '3', '4'],
                        type = str)

    parser.add_argument('-fr','--fit_exp_rmse', default='1', 
                            help='which method to use for fitting exponential part\n1. A*exp-B*t\2. lin', 
                            choices = ["1", "2"],
                            type = str)

    parser.add_argument('-fd','--fit_exp_diff', default='1', 
                        help='which method to use for fitting exponential part\n1. A*exp-B*t\2. lin', 
                        choices = ["1", "2"],
                        type = str)

    parser.add_argument('-rs','--rwc_sup', default=80, 
                        help='upper percentage boundary of RWC [0 .. 100]',                         
                        type=float_range(0,100))
    
    parser.add_argument('-ri','--rwc_inf', default=50, 
                        help='lower percentage boundary of RWC [0 .. 100]',                         
                        type=float_range(0,100))


    args = parser.parse_args()



    
    time_col = args.time_col
    sample_id = args.sample_id
    yvar = args.yvar
    temp = args.temp
    rh = args.rh
    patm = args.patm
    area = args.area

    iter_n = args.iter_n
    alpha = args.alpha
    epsilon = args.epsilon
    diff_method = args.diff_method

    transfo_rmse = args.transfo_rmse
    transfo_diff = args.transfo_diff

    fit_exp_rmse = args.fit_exp_rmse
    fit_exp_diff = args.fit_exp_diff

    rwc_inf = args.rwc_inf
    rwc_sup = args.rwc_sup

    assert rwc_inf+rwc_sup < 100, 'sum of rwc boundaries should be < 100'
    assert rwc_inf < rwc_sup, 'rwc boundary inf should be < to rwc boundary sup'

    print('\n')
    print('Parametrizable parameters are :')
    print('--------------------------------\n')

    for arg in vars(args):
        print(arg,' : ', getattr(args, arg))
    
    print('\n')


    parse_folder = ParseTreeFolder(time_col = time_col,
                                   sample_id = sample_id,
                                   yvar = yvar,
                                   temp = temp,
                                   rh = rh,
                                   patm = patm,
                                   area = area,
                                   iter_n = iter_n,
                                   alpha = alpha,
                                   epsilon = epsilon,
                                   diff_method = diff_method,
                                   transfo_rmse = transfo_rmse,
                                   transfo_diff = transfo_diff,
                                   fit_exp_rmse = fit_exp_rmse,
                                   fit_exp_diff = fit_exp_diff,
                                   rwc_inf = rwc_inf,
                                   rwc_sup = rwc_sup
                                  )
    parse_folder.parse_folder()
    
    parse_folder.run()

