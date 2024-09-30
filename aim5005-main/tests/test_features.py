from aim5005.features import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import unittest
from unittest.case import TestCase

### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    # TODO: Add a test of your own below this line
    def test_standard_scaler_zero_variance(self):
        scaler = StandardScaler()
        data = [[1, 1], [1, 1], [1, 1], [1, 1]]
        expected = np.array([0., 0.])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected, result)
        
    #This code implementation is based on references from:
    # https://www.geeksforgeeks.org/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/ and  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html for logic and structure assistance.
    def  test_initialize_label_encoder(self):
        encoder = LabelEncoder()
        assert isinstance(encoder, LabelEncoder), "encoder is not a LabelEncoder object"
    
    #to show that it can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels i have used variables
    def test_label_encoder_fit(self):
        encoder = LabelEncoder()
        data = ['a', 'b', 'c', 'a', 'b', 'c']
        expected = np.array(['a', 'b', 'c'])
        encoder.fit(data)
        assert (encoder.classes_ == expected).all(), "encoder fit does not return expected classes. Expect {}. Got {}".format(expected, encoder.classes_)

    def test_label_encoder_transform(self):
        encoder = LabelEncoder()
        data = ['a', 'b', 'c', 'a', 'b', 'c']
        expected = np.array([0, 1, 2, 0, 1, 2])
        encoder.fit(data)
        result = encoder.transform(data)
        assert (result == expected).all(), "encoder transform does not return expected values. Expect {}. Got {}".format(expected, result)
    
    def test_label_encoder_fit_transform(self):
        encoder = LabelEncoder()
        data = ['a', 'b', 'c', 'a', 'b', 'c']
        expected = np.array([0, 1, 2, 0, 1, 2])
        result = encoder.fit_transform(data)
        assert (result == expected).all(), "encoder fit_transform does not return expected values. Expect {}. Got {}".format(expected, result)

   
if __name__ == '__main__':
    unittest.main()