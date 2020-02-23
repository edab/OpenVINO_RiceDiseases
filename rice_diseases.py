import argparse
import time

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
RICE_MODEL = "models/rice_model.xml"
RICE_WEIGHTS = "models/rice_model.bin"

HEALTY_IMG="dataset/rice-diseases-image-dataset/LabelledRice/Labelled/Healthy/IMG_3108.jpg"
BROWNSPOT_IMG="dataset/rice-diseases-image-dataset/LabelledRice/Labelled/BrownSpot/IMG_20190420_185845.jpg"
HISPA_IMG="dataset/rice-diseases-image-dataset/LabelledRice/Labelled/Hispa/IMG_20190419_095802.jpg"
LEAFBLAST_IMG="dataset/rice-diseases-image-dataset/LabelledRice/Labelled/LeafBlast/IMG_3009.jpg"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")

    # Create the descriptions for the commands
    p_desc = "The location of the input file (default: random image from dataset)"
    i_desc = "The type of input: 0 = camera, 'IMAGE', 'VIDEO' (default: 'IMAGE')"
    d_desc = "Target device: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU (default: 'CPU')"

    # Create the arguments
    parser.add_argument("-p", help=i_desc, default=HEALTY_IMG)
    parser.add_argument("-i", help=i_desc, default='IMAGE')
    parser.add_argument("-d", help=d_desc, default='CPU')
    args = parser.parse_args()

    return args

def load_network():

    print('Loading Rice Diseases Model....')

    # Initialise the class
    Network = IENetwork(model=RICE_MODEL, weights=RICE_WEIGHTS)

    # Get Input Layer Informationoutput_imag    image = cv.imread(arguments.input)
    RiceDiseaseInputLayer = next(iter(Network.inputs))
    print("Rice Disease Input Layer: ", RiceDiseaseInputLayer)
    print(Network.inputs)

    # Get Output Layer Informationargs
    RiceDiseaseOutputLayer = next(iter(Network.outputs))
    print("Rice Disease Output Layer: ", RiceDiseaseOutputLayer)

    # Get Input Shape of Model
    RiceDiseaseInputShape = Network.inputs[RiceDiseaseInputLayer].shape
    print("Rice Disease Input Shape: ", RiceDiseaseInputShape)

    # Get Output Shape of Model# Read Image
    RiceDiseaseOutputShape = Network.outputs[RiceDiseaseOutputLayer].shape
    print("Rice Disease Output Shape: ", RiceDiseaseOutputShape)

    # Get Shape Values for Face Detection Network
    N, C, H, W = Network.inputs[RiceDiseaseInputLayer].shape

    return N, C, H, W, Network

def run_app(args):

    # Load IECore Object
    OpenVinoIE = IECore()
    print("Available Devices: ", OpenVinoIE.available_devices)

    # Load CPU Extensions if Necessary
    if args.d == 'CPU':
        print('Loading CPU extensions....')
        OpenVinoIE.add_extension(CPU_EXTENSION, 'CPU')

    # Load Networkprint(N, C, H, W)

    N, C, H, W, Network = load_network()

    # Load Executable Network
    ExecutableNetwork = OpenVinoIE.load_network(network=Network, device_name=args.d)

    if args.i == "IMAGE":

        # Read Image
        image = cv2.imread(args.p)

        # Pre-process Image
        resized = cv2.resize(image, (W, H))
        resized = resized.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        print("resized shape: {}".format(resized.shape))
        input_image = resized.reshape((C, H, W))

        # Start Inference
        start = time.time()
        results = ExecutableNetwork.infer(inputs={'x.1': input_image})
        end = time.time()
        inf_time = end - start
        print('Inference Time: {} Seconds'.format(inf_time))

        fps = 1./(end-start)
        print('Estimated FPS: {} FPS'.format(fps))
        print(results['562'][1])
        index = np.where(results['562'][1] == np.amax(results['562'][1]))
        res = np.asscalar(index[0])

        fh = image.shape[0]
        fw = image.shape[1]

        # Write Information on Image
        imS = cv2.resize(image, (960, 540))                    # Resize image
        text = 'FPS: {}, INF: {}, IDX: {}'.format(round(fps, 2), round(inf_time, 2), res)
        cv2.putText(imS, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 125, 255), 1)

        cv2.namedWindow("OpenVINO Rice Diseases")        # Create a named window
        cv2.moveWindow("OpenVINO Rice Diseases", 40,30)  # Move it to (40,30)
        cv2.imshow("OpenVINO Rice Diseases", imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.i == "VIDEO" or args.i == 0:

        # Generate a Named Window
        cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Window', 800, 600)

        # Get and open video capture
        cap = cv2.VideoCapture()
        cap.open(args.i)
        has_frame, frame = cap.read()

        # Get frame size
        fh = frame.shape[0]
        fw = frame.shape[1]
        print('Original Frame Shape: ', fw, fh)

        infer_on_video(args, N, C, H, W, Network, ExecutableNetwork, cap)

        # Release the capture and destroy any OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    '''
    Start the OpenVINO Rice Disease diagnosing app
    '''
    args = get_args()

    run_app(args)
