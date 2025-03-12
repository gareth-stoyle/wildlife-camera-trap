# wildlife-camera-trap
A Raspberry Pi wildlife camera trap  which captures footage and asynchronously feeds frames into the ML animal detection pipeline, running on Hailo AI architecture.

This makes use of a model I have compiled for use on the Hailo AI chip, original model is from the paper "Recognizing European mammals and birds in camera trap images using convolutional neural networks" (Schneider et al., 2023).

## Example
Here is an example of the pipeline being run on sample footage (while I find the time to actually deploy the trap somewhere).


![fox detection](fox_footage.gif)

## Equipment
- Raspberry Pi 5
- Hailo 8L AI kit
- RPi NoIR camera
- RPi Battery pack and holder
- IR LED light (and battery pack)
- Trap enclosure (such as https://thepihut.com/products/naturebytes-wildlife-camera-case-shell-only)
