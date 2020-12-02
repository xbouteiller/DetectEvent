#!/home/xavier/anaconda3/bin/python
if __name__=="__main__":


    #%% Dependencies
    import os
    from LeafConductance.leafconductance import ParseFile, ParseTreeFolder

    parse_folder = ParseTreeFolder()
    parse_folder.parse_folder()
    
    parse_folder.run()

