[RiceDiseases]: images/RiceDiseases.png "Rice Diseases"
[Screenshot]: images/InferScreenshot.png "Current app infer screenshot"

# OpenVINO Rice Disease Diagnoser

The aim of this project is to create a ***Rice Disease Diagnoser portable device***, based on the _[Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/en-us/openvino-toolkit)_, the _[Raspberry Pi 3](https://www.raspberrypi.org/products/raspberry-pi-3-model-b/)_ board, the _[Intel® Movidius™ Neural Compute Stick 2](https://software.intel.com/en-us/neural-compute-stick)_.

>This project was created as a final showcase activity for the [Intel® Edge AI Scholarship Program](https://www.udacity.com/scholarships/intel-edge-ai-scholarship), which focuses on ***IoT*** and ***AI at the Edge*** using the Intel® OpenVINO™ toolkit.

## Project description

Although the _OpenVINO™ Toolkit_ comes with a large number of pre-trained and optimized models, we have chosen to develop our own customized model and optimize it for execution on an Intel CPU, by following these steps:

- Train a model using PyTorch
- Convert the trained model to the ONNX format
- Use the Intel OpenVINO toolkit to optimize the model for _IoT AI Edge application_

### What is AI at the Edge?

The ***Edge*** means local (or near local) processing, as opposed to just anywhere in the cloud. This can be an actual local device like a smart refrigerator, or servers located as close as possible to the source (i.e. servers located in a nearby area instead of on the other side of the world).

The ***Edge*** can be used where low latency is necessary, or where the network itself may not always be available. The use of it can come from a desire for real-time decision-making in certain applications.

### What is the OpenVINO™ Toolkit?

The OpenVINO Toolkit facilitates developing applications and solutions that emulate human vision, enabling:

- Deep learning inference from edge to cloud
- Accelerate AI workloads (Computer Vision, Speech, Language Processing)
- Support for heterogeneous execution across Intel® architecture (CPU, GPU, VPU, FPGA)
- Speeds up time-to-market via library of functions and pre-optimized kernels
- Optimized calls for OpenCV or OpenCL™ kernels

### Dataset used

The **Rice Diseases Image Dataset** used for this project is available at [Kaggle](https://www.kaggle.com/minhhuy2810/rice-diseases-image-dataset/download), thanks to Huy Minh Do, and contains a labeled set of four type of images

![RiceDiseases]

### Transfer Learning with PyTorch Framework

The size of the dataset is 3355. The dataset was split into a training dataset 80% (2684), validation 10% (335), and testing 10% (336).

The first attempt was using a DenseNet-201 (with a learning rate of 0.01) and freezing all the weights from the pretrained network. The accuracy was very poor, around 50%. Then, with another model, VGG16 (learning rate 0.01) with frozen weights from the pretrained network, accuracy was at 39%. ResNet-50 showed the same poor accuracy of 50%. This showed that the dataset is very different from original training image database for these pretrained models. To improve the accuracy, we would need to retrain the network from scratch with randomly initialized weights.

```
# Freeze training for all 'features' layers
for param in model_transfer.features.parameters():
    param.requires_grad=True
```
Setting ```param.requires_grad to True``` increased the DenseNet-201 accuracy to 80%. In this project, we have chosen the model trained using the GoogleNet architecture, as it has the highest accuracy of 97%.

### Conversion from PyTorch to ONNX to OpenVINO Intermediate Representation

The PyTorch model (model_transfer.pt) was first converted to ONNX format using ```torch.onnx.export```. One learning from this process was to match the tensor dimensions exactly as we would expect on the inference end. We had to pick a batch size of 1, instead of a higher number because we were feeding in one input image at a time. This would be different for a video stream.

The resulting ONNX model was then converted to _OpenVINO Intermediate Representation (IR)_ using the _OpenVINO Model Optimizer_

### Application Inference

Once the model was successfully converted into the _OpenVINO Toolkit Intermediate Representation (IR)_, the development of the inference code was smooth.

The only **critical** point to be aware of is that the input image transformations used during the training process must be now applied using OpenCV on the input frames during inference, enabling the optimized model to perform the same way as it was trained.

## Current application Screenshot

The current application is able to infer on single images and write the class identified and the inference time in ms.

![Screenshot]

## Quickstart

## Usage

For running the application, you can use the following command:

```bash
source /opt/intel/openvino/bin/setupvars.sh

python3 rice_diseases.py [-h] [-p P] [-i I] [-d D]

The optional arguments are:
  -h, --help  show this help message and exit
  -p P        The location of the input file (default: random image from dataset)
  -i I        The type of input: 0 = camera, 'IMAGE', 'VIDEO' (default: 'IMAGE')
  -d D        Target device: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA, CPU (default: 'CPU')
```

## Future Extensions

In this work, we demonstrated that we can pick a domain-specific problem (for which there might not exist pretrained models within the OpenVINO toolkit), train a model to help address the problem, and then deploy an app that infers field cases using the OpenVINO Toolkit. Our proof-of-concept uses the CPU, and static images.

The first natural extension to this endeavor is to enhance our app to accept an input video stream, and classify the different diseases for a rice plant on a farm.

The backend work for this would involve enhancing the inference code to process video streams, and also an update on the training side to add a "not rice" category to be able to classify all the frames that do not have a rice plant in them. After that, the number of diseased plants, the kinds of disease, and more such statistics can be reported out.

However, the real value of OpenVINO and its inference acceleration can be tapped when we deploy this on a portable Raspberry-Pi system, that can infer diseased-or-not in real-time, from an input stream through an onboard camera, and signal (with a beep, or a light flash), when diseased plants are found. The system would also log all such events.

Such a system - a Rice Health Inspector - can be operated by a live worker or a robot on a farm to quickly diagnose the health of the overall harvest, and take measures to prune/quarantine. Location data at the sites of disease can also be recorded (as might be in the case of a robot doing an inspection pass through the entire farm), which can then be followed up by expert farmers to properly triage the diseased plants.

Such an application can hugely improve the productivity and efficiency of rice plant farming, which is a staple food for many in the world.

## References
- [Kaggle rice diseases dataset](https://www.kaggle.com/minhhuy2810/rice-diseases-image-dataset)
- [OpenVINO installation guide](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html)
- [PyTorch to ONNX conversion](https://michhar.github.io/convert-pytorch-onnx/)
