__author__ = 'eremeykin'
import unittest
import configparser
import numpy as np
from time import time
from eclustering.ik_means import ik_means


class TestIKMeans(unittest.TestCase):

    @staticmethod
    def setUpClass():
        TestIKMeans.config = configparser.ConfigParser()
        TestIKMeans.config.read('config.ini')
        TestIKMeans.settings = TestIKMeans.config['settings']
        TestIKMeans.data_folder = TestIKMeans.config['settings']['DataFolder'].rstrip('/')
        TestIKMeans.verify_folder = TestIKMeans.config['settings']['VerifyFolder'].rstrip('/')

    def abstract_test(self, test):
        test_data = TestIKMeans.data_folder + test
        verify_data = TestIKMeans.verify_folder + test
        tdata = np.loadtxt(test_data)
        vdata = np.loadtxt(verify_data)
        # np.random.shuffle(tdata)
        start = time()
        output = ik_means(tdata)
        end = time()
        assert np.array_equal(output, vdata)
        print(test + ' N=' + '{:5d}'.format(len(tdata)) + ' finished in ' + '{:0.4f}'.format(end - start) + ' sec.')

    def test_1(self):
        test = '/ikmeans_test1.dat'
        self.abstract_test(test)

    def test_2(self):
        test = '/ikmeans_test2.dat'
        self.abstract_test(test)

    def test_3(self):
        test = '/ikmeans_test3.dat'
        self.abstract_test(test)

    def test_4(self):
        test = '/ikmeans_test4.dat'
        self.abstract_test(test)




