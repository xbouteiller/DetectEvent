#!/home/xavier/anaconda3/bin/python
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
                                   diff_method = diff_method
                                  )
    parse_folder.parse_folder()
    
    parse_folder.run()

