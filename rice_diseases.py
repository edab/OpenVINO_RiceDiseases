import argparse
import time

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

RICE_MODEL = "models/rice_model.xml"
RICE_WEIGHTS = "models/rice_model.bin"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")

    # Create the descriptions for the commands
    i_desc = "The location of the input file (default: 'dataset/biisc/videos/S003_M_COUG_WLK_FCE.avi')"
    d_desc = "Target device: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU (default: 'CPU')"

    # Create the arguments
    parser.add_argument("-i", help=i_desc, default=0)
    parser.add_argument("-d", help=d_desc, default='CPU')
    args = parser.parse_args()

    return args

def run_app(args):
    print("App stub")

if __name__ == "__main__":
    '''
    Start the OpenVINO Rice Disease diagnosing app
    '''
    args = get_args()

    run_app(args)
