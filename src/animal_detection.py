from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams, 
                            FormatType)
from logger import customLogger
import numpy as np
import tensorflow as tf
import time
import yaml

logger = customLogger("animal_detection", "outputs/app.log", debug=False)

with open("src/model/labels.yaml") as f:
    LABELS = yaml.load(f, yaml.SafeLoader)

try:
    # Run hailo inference
    TARGET = VDevice()
    MODEL_PATH = "src/model/EfficientNetV2_quantized.hef"
    MODEL = HEF(MODEL_PATH)
    # Configure network groups
    CONFIGURE_PARAMS = ConfigureParams.create_from_hef(hef=MODEL, interface=HailoStreamInterface.PCIe)
    NETWORK_GROUPS = TARGET.configure(MODEL, CONFIGURE_PARAMS)
    NETWORK_GROUP = NETWORK_GROUPS[0]
    NETWORK_GROUP_PARAMS = NETWORK_GROUP.create_params()
    # Create input and output virtual streams params
    INPUT_VSTREAMS_PARAMS = InputVStreamParams.make(NETWORK_GROUP, format_type=FormatType.FLOAT32)
    OUTPUT_VSTREAMS_PARAMS = OutputVStreamParams.make(NETWORK_GROUP, format_type=FormatType.UINT8)
    USE_HAILO = True
except:
    # Run tensorflow inference (slower)
    USE_HAILO = False
    MODEL_PATH = "src/model/tf/EfficientNetV2"
    MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

def detect_animal_hailo(img):
    # Preprocess image for inference
    img = np.expand_dims(img, axis=0).astype(np.float32)

    start_time = time.time()
    logger.debug("[DEBUG]: Running inference on Hailo")
    with InferVStreams(NETWORK_GROUP, INPUT_VSTREAMS_PARAMS, OUTPUT_VSTREAMS_PARAMS) as infer_pipeline:
        with NETWORK_GROUP.activate(NETWORK_GROUP_PARAMS):
            preds = infer_pipeline.infer(img)['EfficientNetV2/softmax1'][0]

    logger.debug(f"Prediction time: {time.time() - start_time:.4f} seconds")

    index, confidence = np.argmax(preds), np.max(preds)
    confidence = (confidence / 256) * 100
    species = LABELS['species'][index]
    return species, confidence

def detect_animal_tf(img):
    # Preprocess image for inference
    img = np.expand_dims(img, axis=0).astype(np.float32)

    start_time = time.time()
    logger.debug("[DEBUG]: Running inference on Tensorflow")
    preds = MODEL.predict(img)

    logger.debug(f"Prediction time: {time.time() - start_time:.4f} seconds")

    index, confidence = np.argmax(preds), np.max(preds)
    confidence = (confidence / 256) * 100
    species = LABELS['species'][index]
    return species, confidence

if USE_HAILO:
    detect_animal = detect_animal_hailo
else:
    detect_animal = detect_animal_tf