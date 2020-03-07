import argparse
import time
import os, random
import cv2
import numpy as np
from glob import glob

from openvino.inference_engine import IENetwork, IECore

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

RICE_MODEL = "models/rice_model_1_batch.xml"
RICE_WEIGHTS = "models/rice_model_1_batch.bin"

DATADIR="dataset/rice-diseases-image-dataset/LabelledRice/Labelled/"

IMGS = ["dataset/rice-diseases-image-dataset/LabelledRice/Labelled/Healthy/IMG_3108.jpg",
        "dataset/rice-diseases-image-dataset/LabelledRice/Labelled/BrownSpot/IMG_20190420_185845.jpg",
        "dataset/rice-diseases-image-dataset/LabelledRice/Labelled/Hispa/IMG_20190419_095802.jpg",
        "dataset/rice-diseases-image-dataset/LabelledRice/Labelled/LeafBlast/IMG_3009.jpg"]

def get_dataset():

    # Check the number of images present
    images = glob(os.path.join(DATADIR, '*/*.jpg'))
    tot_images = len(images)
    print('Total images:', tot_images)

    # Populate the class name list
    tot_images = 3355
    im_cnt = []
    class_names = []
    print('{:18s}'.format('Class'), end='')
    print('Count')
    print('-' * 24)
    for folder in os.listdir(os.path.join(DATADIR)):
        folder_num = len(os.listdir(os.path.join(DATADIR, folder)))
        im_cnt.append(folder_num)
        class_names.append(folder)
        print('{:20s}'.format(folder), end=' ')
        print(folder_num)
        if (folder_num < tot_images):
            tot_images = folder_num
            folder_num = folder

    num_classes = len(class_names)
    print('Total number of classes: {}'.format(num_classes))

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")

    # Pick input files
    get_dataset()

    # Create the descriptions for the commands
    p_desc = "The location of the input file (default: random image from dataset)"
    i_desc = "The type of input: 0 = camera, 'IMAGE', 'VIDEO' (default: 'IMAGE')"
    d_desc = "Target device: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU (default: 'CPU')"

    # Create the arguments
    parser.add_argument("-p", help=p_desc, default=IMGS[random.randint(0,3)])
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

def pre_process(image, C, W, H):
    # Original transformation is:
    #   transform = cvtransforms.Compose([
    #     cvtransforms.Resize(225),
    #     cvtransforms.CenterCrop(224),
    #     cvtransforms.ToTensor(),
    #     cvtransforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #   ])

    print("Input image transformation")

    print('  - Size: {}'.format(image.shape))
    print('  - Type: {}'.format(type(image)))
    print('  - Type: %s' % image.dtype)
    print('  - Min: %.3f' % (image.min()))
    print('  - Max: %.3f' % (image.max()))

    print("  Normalizing image")

    # Convert from integers to floats
    normalized = image.astype('float32')

    # Normalize using the same values used on Tensorflow for train the model
    normalized /= 255.0
    normalized -= [0.485, 0.456, 0.406]
    normalized /= [0.229, 0.224, 0.225]

    print("  Resize to {} x {}".format(W, H))

    # Resize the model
    resized = cv2.resize(normalized, (W, H))

    print("  Reshaping: {}".format(resized.shape))

    # Change data layout from HWC to CHW
    resized = resized.transpose((2, 0, 1))

    return resized.reshape((C, H, W))

def run_app(args):

    classes = ['Brown spot', 'Hispa', 'Leaf blast', 'Healty']

    random.seed(time.time())

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
        input_image = pre_process(image, C, W, H)

        # Start Inference
        start = time.time()
        results = ExecutableNetwork.infer(inputs={'x.1': input_image})
        index = results['562'].argmax()
        end = time.time()
        inf_time = (end - start) * 1000
        print('Inference Time: {} ms'.format(inf_time))

        print('Results: {}'.format(results['562']))
        print('Type: {}'.format(type(results['562'])))
        print('Min: {}'.format(results['562'].argmin()))
        print('Max: {}'.format(results['562'].argmax()))
        print('Val: \'{}\' [{}]'.format(classes[index], index))

        fh = image.shape[0]
        fw = image.shape[1]

        # Write Information on Image
        imS = cv2.resize(image, (960, 540))                    # Resize image
        text = 'Inf: {} ms, Disease: {}'.format(round(inf_time, 1), classes[index])
        cv2.putText(imS, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 125, 255), 1)

        # Show original image and prediction
        cv2.namedWindow("OpenVINO Rice Diseases")         # Create a named window
        cv2.moveWindow("OpenVINO Rice Diseases", 40, 30)  # Move it to (40,30)
        cv2.imshow("OpenVINO Rice Diseases", imS)
        #cv2.imshow("OpenVINO Rice Diseases", input_image.transpose((1,2,0)))
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
