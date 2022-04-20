import pandas as pd

def read_csv_as_pd(filename):
    """
    Read a csv file and convert to a pandas dataframe

    :param filename: String of path and filename
    :returns: pandas dataframe containing the data in the csv file
    """
    data = pd.read_csv(filename)
    return data
