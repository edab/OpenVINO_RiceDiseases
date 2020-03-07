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

### Transfer Learning with PyTorch Framework

A PyTorch CNN model using GoogleNet to classify rice diseases. Total dataset is 3355 images. It is splitted into 80% training, 10% validation and 10% testing dataset. The testing accuracy using GoogleNet is 97%.

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

### Model developed

## Current application Screen shot

The current application is able to infer on single images and write the class identified and the inter time in ms.

![Screenshot]

## Quickstart



## References
- [Kaggle rice diseases dataset](https://www.kaggle.com/minhhuy2810/rice-diseases-image-dataset)
- [OpenVINO installation guide](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html)
