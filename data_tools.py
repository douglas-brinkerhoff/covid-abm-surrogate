import h5py  
import torch
import numpy as np
from torch.utils import data
import json
from sklearn.preprocessing import normalize
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.nn as nn


# creates the dataset objects needed for training and testing
# splits a dataset into training and test data and does some preprocessing
class hdf5Dataset_init:
    def __init__(self, hdf5_filename, split=.3):
        self.hdf5_size = 0
        self.train_indices = []
        self.file_name = hdf5_filename
        self.index_hdf5 = [] 
        
        # min-max of data, min should always be zero so note stored
        self.max_array= None
        self.min_array = None
        self.initialize_indices()
        self.get_normalize_high_low()
    
    def initialize_indices(self):
        with h5py.File(self.file_name, 'r') as f:
            for gen in f.values():
                for trial in gen.values():
                    # index trial name 
                    self.index_hdf5.append(trial.name)
                    
                    # increments total number of data by number of datasets in specific trial
                    data_size = len(trial['data/positive_tests_total/all_runs'][...])
                    if data_size != 5:
                        print(data_size)
                        
                    # get total data inside
                    self.hdf5_size+=len(trial['data/positive_tests_total/all_runs'][...])
    
    # returns training and testing dataset for use
    def split_datasets(self, split, split_type):
        test_ind = []
        training_ind = []
           
           
        if split_type == "fully_random":
            # create list of indices
            indices = np.arange(self.hdf5_size)
            # shuffle that list
            np.random.shuffle(indices)
            print('split ind ', int(split*self.hdf5_size))
            test_ind, training_ind = np.split(indices, [int(split*self.hdf5_size), self.hdf5_size]#split*filesize gives size of test portion
                                             )[:2] # np.split returns 3rd empty arr so must ignore that
        elif split_type == 'parameter':
            indices = np.arange(self.hdf5_size//5) # get number of unique paramter sets
            np.random.shuffle(indices)
            test_num, training_num = np.split(indices, [int(split*self.hdf5_size//5), self.hdf5_size//5]#split*filesize gives size of test portion
                                             )[:2] # np.split returns 3rd empty arr so must ignore that
           
            # iterate through test parameter sets
            for i in test_num:
                # get each of the 5 indices that are part of that paramter set
                for j in range(5):
                    test_ind.append((i*5)+j) # i+j gives the indice of each data set in that paramter
           
            for i in training_num:
                # get each of the 5 indices that are part of that paramter set
                for j in range(5):
                    training_ind.append((i*5)+j) # i+j gives the indice of each data set in that paramter
        else:
            print('select either \' parameter \' or \' fully_random \' for split_type')
            return
       
        #get device name
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # TRAINING
        training_inp = []
        training_output = []
       
        for ind in training_ind:
            combine = self.combine_data(ind)
           
            # only happens with bad output i.e -1, -1
            if combine is None:
                continue#append new item
           
           
            training_inp.append(combine[0])
            training_output.append(combine[1])
           
            #convert to numpy
        training_output = np.array(training_output,dtype=np.float32)
        training_inp = np.array(training_inp,dtype=np.float32)
       
        # convert output to log form
        #training_output = np.log(training_output, where=(training_output!=0))

       
        #convert from numpy to torch
        training_inp = torch.from_numpy(training_inp)
        training_output = torch.from_numpy(training_output)
       
        # convert to device type
        training_output = training_output.to(device)
        training_inp = training_inp.to(device)

        trainingTensor = TensorDataset(training_inp, training_output)
       
         # TESTING
        testing_inp = []
        testing_output = []
       
        for ind in test_ind:
            combine = self.combine_data(ind)
           
            #check if datset is bad and pass if yes
            if combine is None:
                continue
            #append new item
            testing_inp.append(combine[0])
            testing_output.append(combine[1])
           
            #convert to numpy
        testing_output = np.array(testing_output,np.float32)
        testing_inp = np.array(testing_inp,dtype=np.float32)

        # get the log version of the output
        #testing_output = np.log(testing_output, where=(testing_output!=0))
       
        #convert from numpy to torch
        testing_inp = torch.from_numpy(testing_inp)
        testing_output = torch.from_numpy(testing_output)
       
        # convert to device type and float
        testing_output = testing_output.to(device)
        testing_inp = testing_inp.to(device)
       
        testingTensor = TensorDataset(testing_inp, testing_output)
       
        #convert to float for neural network
        return trainingTensor, testingTensor  
        
    def combine_data(self, indice):
        target_group = self.index_hdf5[indice//5] # get larger group name which contrains 5 datasets
        inner_index = indice % 5 # get index of group within the 5
        data = []
        with h5py.File(self.file_name, 'r') as F:
            trial = F[target_group] # get the specified indexs trial group
            params = json.loads(trial['parameters'][...].tolist()) # get the parameters
            cleaned_dict = self.transform_dict(params)
            run_data = []
            for item in trial['data'].values(): # get parameters objects
                data = item['all_runs'][...][inner_index]
                
                # bad dataset
                '''if max(data) < 1:
                    return
                '''
                run_data.append(item['all_runs'][...][inner_index]) # get indexed runs data 
                
            np_run = np.array(run_data) 
            
            return cleaned_dict, run_data
                
                 
    # get high and lows for parameters to allow for min_max standardizationk
        
    # do some preprocessing on dictionary
    def transform_dict(self,dictionary, normalize=False):
        # remove unnessarry items from the parameter dictionarys
        bad_params = ['test','verbose','attendance_bins','scenario_name','parameter_checking',
                      'run_days', 'daily_outside_cases','removed_cohorts']
        for param in bad_params:
            dictionary.pop(param, 'None')
            
        #print(dictionary.keys())
        # convert vlaues to numpy array format
        dict_vals = np.array(list(dictionary.values()))
        
        
        # normalize the inputs
        if normalize:
            dict_vals = (np.subtract(dict_vals, self.min_array))/self.max_array
            
        
        return dict_vals
    
    def get_normalize_high_low(self):
        random_sample = np.random.randint(0, high=self.hdf5_size, size=512)
        
        
        array = []
        # find min max for randomzied items from dataset
        with h5py.File(self.file_name, 'r') as F:

            for rand in random_sample:
                
                dict_P = json.loads(F[self.index_hdf5[rand//5]]['parameters'][...].tolist()) # get param dictionary
                values = self.transform_dict(dict_P, normalize=False)
                # first iteration
                if rand == random_sample[0]:
                    array = values
                else:
                    array = np.vstack((array, values))

            self.max_array = np.amax(array, axis=0)
            self.min_array = np.amin(array, axis=0)


def normalize_sets(training,testing,X_mean=None,X_std=None):
    X_averaged = torch.stack([torch.mean(block,axis=0) for block in torch.split(training.tensors[0],5)])
    y_averaged = torch.stack([torch.mean(block,axis=0) for block in torch.split(training.tensors[1],5)])

    index = (X_averaged[:,0]==1)*(X_averaged[:,1]==1)*(X_averaged[:,-1]==1)
    X_averaged = X_averaged[index]
    y_averaged = y_averaged[index]
    X_averaged = X_averaged[:,2:-1]

    if X_mean is None:
        X_mean = X_averaged.mean(axis=0)
    if X_std is None:
        X_std = X_averaged.std(axis=0)
    
    X_averaged -= X_mean
    X_averaged /= X_std

    training_ = TensorDataset(X_averaged,y_averaged)
    try:
        X_averaged = torch.stack([torch.mean(block,axis=0) for block in torch.split(testing.tensors[0],5)])
        y_averaged = torch.stack([torch.mean(block,axis=0) for block in torch.split(testing.tensors[1],5)])

        index = (X_averaged[:,0]==1)*(X_averaged[:,1]==1)*(X_averaged[:,-1]==1)
        X_averaged = X_averaged[index]
        y_averaged = y_averaged[index]
        X_averaged = X_averaged[:,2:-1]

        X_averaged -= X_mean
        X_averaged /= X_std
        testing_ = TensorDataset(X_averaged,y_averaged)
    except IndexError:
        testing_ = testing

    return training_,testing_,X_mean,X_std

class Emulator(nn.Module):
    def __init__(self,n_parameters=26,n_eigenfunctions=64,n_hidden_1=128,n_hidden_2=128,n_hidden_3=128,n_hidden_4=128,n_hidden_head=128,n_grid_points=128):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.l_1 = nn.Linear(n_parameters, n_hidden_1)
        self.norm_1 = nn.LayerNorm(n_hidden_1)
        self.dropout_1 = nn.Dropout(p=0.0)
        self.l_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.norm_2 = nn.LayerNorm(n_hidden_2)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.l_3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.norm_3 = nn.LayerNorm(n_hidden_3)
        self.dropout_3 = nn.Dropout(p=0.2)
        self.l_4 = nn.Linear(n_hidden_3, n_hidden_4)
        self.norm_4 = nn.LayerNorm(n_hidden_3)
        self.dropout_4 = nn.Dropout(p=0.2)
        self.l_5 = nn.Linear(n_hidden_4, n_eigenfunctions)
        self.V_1a = nn.Linear(n_eigenfunctions,n_hidden_head,bias=True)
        self.V_2a = nn.Linear(n_eigenfunctions,n_hidden_head,bias=True)
        self.V_3a = nn.Linear(n_eigenfunctions,n_hidden_head,bias=True)
        
        self.V_1b = nn.Linear(n_hidden_head,n_grid_points,bias=True)
        self.V_2b = nn.Linear(n_hidden_head,n_grid_points,bias=True)
        self.V_3b = nn.Linear(n_hidden_head,n_grid_points,bias=True)
        
        self.dropout_5 = nn.Dropout(p=0.0)
        self.leaky_relu = nn.LeakyReLU(1e-2)

        #self.V_hat = torch.nn.Parameter(V_hat,requires_grad=False)
        #self.F_mean = torch.nn.Parameter(F_mean,requires_grad=False)

    def forward(self, x, add_mean=False):
        # Pass the input tensor through each of our operations

        a_1 = self.l_1(x)
        a_1 = self.norm_1(a_1)
        a_1 = self.dropout_1(a_1)
        z_1 = self.leaky_relu(a_1) 

        a_2 = self.l_2(z_1)
        a_2 = self.norm_2(a_2)
        a_2 = self.dropout_2(a_2)
        z_2 = self.leaky_relu(a_2) + z_1

        a_3 = self.l_3(z_2)
        a_3 = self.norm_3(a_3)
        a_3 = self.dropout_3(a_3)
        z_3 = self.leaky_relu(a_3) + z_2
        
        a_4 = self.l_4(z_3)
        a_4 = self.norm_3(a_4)
        a_4 = self.dropout_4(a_4)
        z_4 = self.leaky_relu(a_4) + z_3
        
        z_5 = self.l_5(z_4)
        z_5 = self.dropout_5(z_5)
        z_6a = self.V_1b(self.leaky_relu(self.V_1a(z_5)))
        z_6b = self.V_2b(self.leaky_relu(self.V_2a(z_5)))
        z_6c = self.V_3b(self.leaky_relu(self.V_3a(z_5)))

        return torch.stack((z_6a, z_6b, z_6c),axis=1)

