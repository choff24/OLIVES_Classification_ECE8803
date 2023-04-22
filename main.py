import subprocess
import os

# This is just a quick runner for all of the models if I wanted to batch run everything at once
subprocess.run(['python', os.getcwd() + '\Train_Autoencoder_3D.py'])
subprocess.run(['python', os.getcwd() + '\Train_CNN_3D.py'])
subprocess.run(['python', os.getcwd() + '\Train_ANN_3D.py'])
subprocess.run(['python', os.getcwd() + '\Train_SKLearn_Models.py'])



