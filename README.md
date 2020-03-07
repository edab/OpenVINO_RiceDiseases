[RiceDiseases]: images/RiceDiseases.png "Rice Diseases"
[Screenshot]: images/InferScreenshot.png "Current app infer screenshot"

# OpenVINO Rice Disease

The aim of this project is to create a ***Rice Diseases Diagnosing portable device***, based on the _[Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/en-us/openvino-toolkit)_, the _[Raspberry Pi 3](https://www.raspberrypi.org/products/raspberry-pi-3-model-b/)_ board, the _[Intel® Movidius™ Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick)_.

>This project was created as a final showcase activity for the [Intel® Edge AI Scholarship Program](https://www.udacity.com/scholarships/intel-edge-ai-scholarship), which focus on ***IoT*** and ***AI at the Edge*** using the Intel® OpenVINO™ toolkit.

## Project description

Although the _OpenVINO™ Toolkit_ comes with a large number of pre-trained and optimized models, we have chosen to develop our own customized model and optimize it for execution on the Raspberry Pi, by following these steps:

- Train a model using PyTorch
- Convert the trained model to the ONNX format
- Use the Intel OpenVINO toolkit for optimize the model for _IoT AI Edge application_

### What is AI at the Edge?

The ***Edge*** means local (or near local) processing, as opposed to just anywhere in the cloud. This can be an actual local device like a smart refrigerator, or servers located as close as possible to the source (i.e. servers located in a nearby area instead of on the other side of the world).

The ***Edge*** can be used where low latency is necessary, or where the network itself may not always be available. The use of it can come from a desire for real-time decision-making in certain applications.

### What is the OpenVINO™ Toolkit?

The OpenVINO Toolkit help developing applications and solutions that emulate human vision, enabling:

- Deep learning inference from edge to cloud
- Accelerates AI workloads (Computer Vision, Speech, Language Processing)
- Support for heterogeneous execution across Intel® architecture (CPU, GPU, VPU, FPGA)
- Speeds up time to market via library of functions and pre-optimized kernels
- Optimized calls for OpenCV or OpenCL™ kernels

### Dataset used

The **Rice Diseases Image Dataset** used for this project is available at [Kaggle](https://www.kaggle.com/minhhuy2810/rice-diseases-image-dataset/download), thanks to Huy Minh Do, and contain a labelled set of four type of images

![RiceDiseases]

### Transfer Learning with PyTorch Framework

The size of the dataset is 3355. The dataset was splitted into training dataset 80%(2684), validation 10%(335), and testing 10%(336).

The first attempt was using DenseNet 201 (learning rate of 0.01) and froze all the weights from the pretrained network. The accuracy was very poor, around 50%. Then, with another model, VGG16 (learning rate 0.01) and froze all the weights from the pretrained network, accuracy was 39%. ResNet 50 showed the same poor accuracy of 50%. It showed that the dataset is totally different from original training image database. To improve the accuracy, we need to retrain the network from scratch with randomly initialized weights.

```
# Freeze training for all 'features' layers
for param in model_transfer.features.parameters():
    param.requires_grad=True
```
Set the ```param.requires_grad to True``` and the DenseNet 201 accuracy increased to 80%. In this competition, we submitted the training model using GoogleNet, as it has the highest accuracy of 97%.

### Application Inference

Once the model was successfully converted into the _OpenVINO Toolkit Intermediate Representation (IR)_, the code for making the actual inference was added smoothly.

The only **critical** point is that the same transformation used for the input images during the training process, should be now applied using OpenCV to the input frames, permitting the optimized model to work on the same way he was trained.

## Current application Screen shot

The current application is able to infer on single images and write the class identified and the inter time in ms.

![Screenshot]

## Quickstart

## Usage

For run the application, you can use the following command:

```bash
source /opt/intel/openvino/bin/setupvars.sh

python3 rice_diseases.py [-h] [-p P] [-i I] [-d D]

The optional arguments are:
  -h, --help  show this help message and exit
  -p P        The location of the input file (default: random image from dataset)
  -i I        The type of input: 0 = camera, 'IMAGE', 'VIDEO' (default: 'IMAGE')
  -d D        Target device: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA, CPU (default: 'CPU')
```

## References
- [Kaggle rice diseases dataset](https://www.kaggle.com/minhhuy2810/rice-diseases-image-dataset)
- [OpenVINO installation guide](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html)
