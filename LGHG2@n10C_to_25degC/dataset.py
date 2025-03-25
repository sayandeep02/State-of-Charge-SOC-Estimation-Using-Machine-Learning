import numpy as np
import pandas as pd
import utils
import sys
import logging
import os



class LGHG2():
    """
    The LGHG2 class is responsible for managing datasets stored in a specific directory structure
    and preparing the data for training, validation, and testing purposes. The class handles
    data loading, checking, and normalization. It is assumed that the data is in CSV format
    and stored in predefined subdirectories ('train', 'val', 'test').

    Attributes:
        _data_dir (str): The root directory containing the data.
        _train_data_dir (str): The directory containing the training data.
        _val_data_dir (str): The directory containing the validation data.
        _test_data_dir (str): The directory containing the test data.
        train_data_df (pd.DataFrame): DataFrame containing the loaded training data.
        val_data_df (pd.DataFrame): DataFrame containing the loaded validation data.
        test_data_df (pd.DataFrame): DataFrame containing the loaded test data.
        _X_train (np.ndarray): Normalized training features.
        _X_val (np.ndarray): Normalized validation features.
        _X_test (np.ndarray): Normalized test features.
        _y_train (np.ndarray): Training targets (labels).
        _y_val (np.ndarray): Validation targets (labels).
        _y_test (np.ndarray): Test targets (labels).
    """

    def __init__(self, data_dir):
        """
        Initializes the LGHG2 class with a specified root directory and sets up paths for
        training, validation, and test data subdirectories. It also logs the contents of
        these directories for inspection.

        Args:
            data_dir (str): The root directory containing the data subdirectories.
        """
        self._data_dir = data_dir
        self._train_data_subdir = 'train'
        self._val_data_subdir = 'val'
        self._test_data_subdir = 'test'
        self._train_data_dir = os.path.join(self._data_dir, self._train_data_subdir)
        self._val_data_dir = os.path.join(self._data_dir, self._val_data_subdir)
        self._test_data_dir = os.path.join(self._data_dir, self._test_data_subdir)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.info(f'{self._data_dir} = {os.listdir(self._data_dir)}')
        logging.info(f'{self._train_data_dir} = {os.listdir(self._train_data_dir)}')
        logging.info(f'{self._val_data_dir} = {os.listdir(self._val_data_dir)}')
        logging.info(f'{self._test_data_dir} = {os.listdir(self._test_data_dir)}')
        
    
    def get_data_dir(self):
        """
        Returns the root data directory.

        Returns:
            str: The root data directory.
        """
        return self._data_dir
    
    def get_train_data_dir(self):
        """
        Returns the training data directory.

        Returns:
            str: The training data directory.
        """
        return self._train_data_dir
    
    def get_val_data_dir(self):
        """
        Returns the validation data directory.

        Returns:
            str: The validation data directory.
        """
        return self._val_data_dir
    
    def get_test_data_dir(self):
        """
        Returns the test data directory.

        Returns:
            str: The test data directory.
        """
        return self._test_data_dir   

    def get_dataset(self):
        """
        Loads and returns the datasets (features and targets) for training, validation,
        and testing after normalization.

        Returns:
            tuple: Normalized training, validation, and test features along with their respective targets.
        """
        return self._get_data()

    
    def _get_data(self):
        """
        Loads the datasets from the respective directories, checks their integrity,
        extracts features and targets, and normalizes the feature data.

        Returns:
            tuple: Normalized training, validation, and test features along with their respective targets.
        """
        self.train_data_df = self._get_df(data_dir=self._train_data_dir, data_ext='.csv')
        self.val_data_df = self._get_df(data_dir=self._val_data_dir, data_ext='.csv')
        self.test_data_df = self._get_df(data_dir=self._test_data_dir, data_ext='.csv')

        self._check_df(df=self.train_data_df, df_name='train_data_df')
        self._check_df(df=self.val_data_df, df_name='val_data_df')
        self._check_df(df=self.test_data_df, df_name='test_data_df')

        logging.info(f'data_columns = {self.train_data_df.columns}')
        features = ['V', 'I', 'Temp', 'V_avg', 'I_avg']
        target = ['SOC']
        logging.info(f'features = {features}')
        logging.info(f'target = {target}')

        self._X_train = self.train_data_df[features].values
        self._X_val = self.val_data_df[features].values
        self._X_test = self.test_data_df[features].values

        self._y_train = self.train_data_df[target].values
        self._y_val = self.val_data_df[target].values
        self._y_test = self.test_data_df[target].values

        return (
            utils.normalize(self._X_train), 
            utils.normalize(self._X_val), 
            utils.normalize(self._X_test), 
             self._y_train, 
            self._y_val, 
            self._y_test
        )


    def get_dfs(self):
        """
        Returns the loaded DataFrames for training, validation, and testing.

        Returns:
            tuple: Training, validation, and test DataFrames.
        """
        return self.train_data_df, self.val_data_df, self.test_data_df

    def get_X_train(self):
        """
        Returns the training features.

        Returns:
            np.ndarray: Training features.
        """
        return self._X_train
        
    def get_X_val(self):
        """
        Returns the validation features.

        Returns:
            np.ndarray: Validation features.
        """
        return self._X_val
        
    def get_X_test(self):
        """
        Returns the test features.

        Returns:
            np.ndarray: Test features.
        """
        return self._X_test   
        
    def get_y_train(self):
        """
        Returns the training targets.

        Returns:
            np.ndarray: Training targets.
        """
        return self._y_train   
        
    def get_y_val(self):
        """
        Returns the validation targets.

        Returns:
            np.ndarray: Validation targets.
        """
        return self._y_val 
        
    def get_y_test(self):
        """
        Returns the test targets.

        Returns:
            np.ndarray: Test targets.
        """
        return self._y_test 

    
    def _get_df(self, data_dir, data_ext):
        """
        Loads CSV files from a specified directory into a single DataFrame.

        Args:
            data_dir (str): The directory to load the data from.
            data_ext (str): The file extension to filter the files (e.g., '.csv').

        Returns:
            pd.DataFrame: A concatenated DataFrame containing all the data from the specified directory.
        """
        listdir = os.listdir(data_dir)
        dfs = []
        for data_name in listdir:
            if data_name.endswith(data_ext):
                data_path = os.path.join(data_dir, data_name)
                data_df = pd.read_csv(data_path)
                dfs.append(data_df)
        df = pd.concat(dfs, ignore_index=True)
        return df

    
    def _check_df(self, df, df_name):
        """
        Checks if a DataFrame is empty and logs the result.

        Args:
            df (pd.DataFrame): The DataFrame to check.
            df_name (str): The name of the DataFrame (used for logging).
        """
        if df.empty:
            logging.info(f'WARNING# {df_name} is empty.')
        else: 
            logging.info(f'{df_name} loaded.')
    