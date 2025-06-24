# Global overview for all experiments

import os
import json
import datetime
import numpy

import sklearn
from experiments import experiment

print(numpy.__version__)

# Set project directory:
DIR = '/Users/brunonicoletta/Developer/CSL Sparse Costs'
if not os.path.isdir(DIR):  # Switch to HPC directory
    DIR = '/data/leuven/341/vsc34195/Experiments/Tests'  # TODO: change per experiment!!
if not os.path.isdir(DIR):  # Switch to Google Colab
    DIR = '/content/drive/My Drive/PhD/Projecten/Cost-sensitive learning/Experiments/Colab/'
assert os.path.isdir(DIR), "DIR does not exist!"

import torch
print('CUDA available?')
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# Specify experimental configuration
#   l1 and l2 not supported simultaneously!
# Extra regarding the sparse and noisy cost matrix setups:
#   If sparse_flag is set to False, a noise will be applied to the cost information with 
#       sparsity_ratio: the ratio betweeen the noisy and clean datasets
#       noise_level: the standard deviation of the applied noise as a ratio of the original values (e.g., 0.1 means 10%)) 
#   If sparse_flag is set to True, the affected cost information will simply set to zero. 
settings = {'class_costs': False,
            'folds': 5,
            'repeats': 1,
            'val_ratio': 0.25,  # Relative to training set only (excluding test set)
            'l1_regularization': False,
            'lambda1_options':  [1e-2, 1e-1], #[0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'l2_regularization': False,
            'lambda2_options': [1e-2, 1e-1], #[0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'neurons_options': [2**8, 2**10], #[2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10],   # Add more for final? original : [2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10] 
            'sparsity_ratio': 0.75,  # Please select a value in [0.0 ; 1.0[  ; good to test [0.8, 0.85, 0.9, 0.95, 0.99]  
            'sparse_flag': False,
            'noise_level': 0.25,
            'Flag_different_validationSetsForTrainingAndTuning': False,   
            'thresholds_comparison': False  # Set by default to Fasle - change if you wish to compare different thresholding appraoch
            }

datasets = {'kaggle credit card fraud': False,
            'kdd98 direct mailing': False,
            'kaggle give me some credit': False, #this one afterwards
            'kaggle telco customer churn': True,
            'uci default of credit card clients': False,
            'uci bank marketing': False,
            'vub credit scoring': False,
            'tv subscription churn': False,
            'kaggle ieee fraud': False
            }

methodologies = {'logit': False,
                 'wlogit': False,
                 'cslogit': False,

                 'net': True,  
                 'wnet': True,
                 'csnet': True,
                 'noisy_wnet': True,
                 'noisy_csnet': True,
                 'sparsity_resilient_csnet': True,
                 'noise_resilient_csnet': True,

                 'boost': False,
                 'wboost': False,
                 'csboost': False
                 }

evaluators = {'LaTeX': True,

              # Cost-insensitive
              'traditional': False,
              'ROC': False,
              'AUC': False,
              'PR': False,
              'H_measure': False,
              'brier': False,
              'recall_overlap': False,
              'recall_correlation': False,

              # Cost-sensitive
              'savings': True,
              'AEC': True,
              'ROCIV': True,
              'PRIV': True,
              'rankings': True,

              # Other
              'time': True,
              'lambda1': True,
              'lambda2': True,
              'n_neurons': True
              }


if __name__ == '__main__':
    
    print('\n\n ***   NEW RUN FOR overview.py    ***  \n\n ')
    print('\n' + datetime.datetime.now().strftime('Experiment started at:  %d-%m-%y  |  %H:%M'))
    print(  *['\n Data selected: ',  datasets, '\n '] )
    print(  *[ '\n  Methodologies: ' ,methodologies, '\n '])

    experiment = experiment.Experiment(settings, datasets, methodologies, evaluators)
    experiment.run(directory=DIR)

    # Create txt file for summary of results
    with open(str(DIR + 'summary.txt'), 'w') as file:
        file.write(str(datetime.datetime.now().strftime('Experiment done at:  %d-%m-%y  |  %H:%M') + '\n'))
        file.write('\nSettings: ')
        file.write(json.dumps(settings, indent=3))
        file.write('\nDatasets: ')
        file.write(json.dumps(datasets, indent=3))
        file.write('\nMethodologies: ')
        file.write(json.dumps(methodologies, indent=3))
        file.write('\nEvaluators: ')
        file.write(json.dumps(evaluators, indent=3))
        file.write('\n\n_____________________________________________________________________\n\n')

    experiment.evaluate(directory=DIR)
