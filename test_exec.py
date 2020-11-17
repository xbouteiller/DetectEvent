#!/home/xavier/anaconda3/bin/python
if __name__=="__main__":


    #%% Dependencies
    import os

    from LeafConductance.essai_code import ParseFile, ParseTreeFolder

    root = os.getcwd()

    parse_folder = ParseTreeFolder()
    parse_folder.parse_folder()
    parse_folder.change_detection()


