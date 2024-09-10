#import unittest and test it
import unittest
from src.data_preparation import load_data, clean_data

class TestDataPreparation(unittest.TestCase):
    def test_load_data(self):
        df = load_data('your_dataset.csv')
        self.assertFalse(df.empty)
    
    def test_clean_data(self):
        df = load_data('your_dataset.csv')
        clean_df = clean_data(df)
        self.assertFalse(clean_df.isnull().values.any())
