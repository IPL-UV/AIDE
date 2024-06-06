import sys
import os

# PATH = "/home/maria/Documents/AIDE_private/AIDE/"
PATH = "/home/miguelangelft/Documents/research/xaida/AIDE_private/AIDE/"

class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )

        self.outnull_file.close()
        self.errnull_file.close()


import warnings
warnings.filterwarnings("ignore")

sys.path.append(PATH)
from utils.setup_config import setup
from main import main

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def set_up_config(config_path):

    config = setup(config_path)
    config['experiment_id'] = config["arch"]["type"]
    # Create experimental folder structure
    if not os.path.isdir(config['save_path']):
        os.mkdir(config['save_path'])
    config['save_path'] = config['save_path'] + config['experiment_id']
    if not os.path.isdir(config['save_path']):
        os.mkdir(config['save_path'])
        
    config["debug"] = 3 #True

    config["evaluation"]["visualization"]["activate"] = True
    if config["task"] == 'Classification':
        config["evaluation"]["characterization"]["activate"] = True
    config["evaluation"]["xai"]["activate"] = True
    
    return config

def test_configuration(config_path):
    print("Running " + config_file +" ...", end=" ")
    config = set_up_config(config_path)
    
    #try:
    #    with suppress_stdout_stderr():
    main(config)

    #    print(f"{bcolors.OKGREEN}[V]{bcolors.ENDC}")
    #except:
    #    print(f"{bcolors.FAIL}[X]{bcolors.ENDC}")

if __name__ == '__main__':
    configs_list = [#PATH + "configs/config_DROUGHT_RUSSIA_3DCONV.yaml",
					#PATH + "configs/config_DROUGHT_RUSSIA_UNET.yaml",
                    #PATH + "configs/config_DROUGHT_RUSSIA_1DLSTM.yaml",
                    #PATH + "configs/config_DROUGHT_RUSSIA_1DTransformer.yaml",
                    #PATH + "configs/config_DROUGHT_RUSSIA_2DTransformer.yaml",
                    #PATH + "configs/config_DROUGHT_RUSSIA_ImpactAssessment_GP.yaml",
                    #PATH + "configs/config_DroughtED_DeepLearning.yaml",
                    #PATH + "configs/config_XAIDA_Floods_IA.yaml",
                    PATH + "configs/config_XAIDA_Floods_IA_GP.yaml"
                    ]
    for config_file in configs_list:
        test_configuration(config_file)
