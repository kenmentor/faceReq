# CHAPTER 5: MODEL DEVELOPMENT AND TRAINING

---

## 5.1 Introduction to Model Development

### 5.1.1 Overview of Model Development Process

This chapter documents the complete process of developing and training the Siamese neural network model used for face verification. The model development process encompasses data collection, preprocessing pipeline design, network architecture implementation, training configuration, the training process itself, and evaluation of the resulting model.

The goal of model development is to create a neural network that can accurately determine whether two facial images show the same person. This requires not only architectural decisions about the network structure but also practical decisions about training data, hyperparameters, and evaluation methodology.

Model development is inherently experimental, requiring iteration and adjustment based on results. The process documented here reflects the final state after experimentation and refinement, but the path to that state involved testing multiple approaches and learning from failures.

### 5.1.2 Development Environment

The model development was performed on a standard laptop computer without GPU acceleration. This constraint influenced many decisions about model architecture and training configuration, favoring efficiency over maximum capacity.

The development environment used Python 3.10 with TensorFlow 2.x for deep learning operations. OpenCV provided image processing capabilities. NumPy handled numerical array operations. Matplotlib generated visualizations of training progress.

The absence of GPU meant that training times were longer than they would be on GPU hardware, but the resulting model is designed to run efficiently on CPU-only systems. This aligns with the project goal of accessibility on standard hardware.

### 5.1.3 Development Workflow

The model development followed a structured workflow that ensured systematic progress and enabled learning from experiments.

The first phase involved data preparation, collecting and organizing training images into a format suitable for the training pipeline. This phase also involved implementing and testing the preprocessing functions that would be applied to all images.

The second phase involved architecture design, creating the neural network structure that would learn the verification task. This phase involved implementing the Siamese architecture with MobileNetV2 backbone and custom comparison layers.

The third phase involved training experimentation, running multiple training sessions with different configurations to find effective hyperparameters. This phase generated the training history and final model weights.

The fourth phase involved evaluation, assessing the trained model on held-out test data to characterize its performance. This phase produced the accuracy metrics and confusion matrices presented in subsequent chapters.

---

## 5.2 Training Data Collection and Preparation

### 5.2.1 Data Sources

The training dataset consists of facial images collected from multiple sources to provide diversity in appearance, lighting, and capture conditions.

The primary data source was images captured using standard webcams under controlled conditions. Multiple sessions were conducted to capture images of volunteers with their consent. Each session captured twenty to thirty images per person under varying conditions.

The secondary data source was existing image datasets made available for research purposes. These datasets provided additional diversity in terms of age, ethnicity, and imaging conditions. Images were selected to complement rather than duplicate the primary source.

The tertiary data source was synthetic augmentation applied during training. Rather than collecting more raw images, the training pipeline augmented existing images to simulate additional variations. This approach is more scalable than raw data collection.

### 5.2.2 Dataset Statistics

The final dataset contains images from approximately twenty to thirty distinct individuals. Each individual contributed between ten and twenty images, resulting in a total of approximately three thousand images.

The images vary in several dimensions that affect recognition difficulty. Some images show frontal faces with neutral expression under good lighting. Others show slight head turns, smiles, or shadows. This variation ensures the model learns robust features rather than overfitting to specific conditions.

The dataset is split into positive pairs (images of the same person) and negative pairs (images of different people). Approximately nine hundred thirty-four positive image pairs and approximately two thousand five hundred sixty-two negative image pairs were generated from the base images.

The training set contains twelve thousand pairs, and the validation set contains three thousand pairs. The split maintains approximately equal numbers of positive and negative pairs in each set.

### 5.2.3 Image Specifications

All images in the dataset meet certain specifications that ensure consistency in the training pipeline.

The images are stored in JPEG format with dimensions of at least two hundred by two hundred pixels before preprocessing. This ensures sufficient resolution for face detection and feature extraction.

The images are captured in color (RGB) rather than grayscale. While the model could process grayscale images, color provides additional information that may improve recognition accuracy.

The images show single, clearly visible faces. Images with multiple faces, heavily occluded faces, or very small faces were excluded from the dataset.

### 5.2.4 Data Organization

The collected images are organized in a directory structure that facilitates training data generation.

Images of the same person are stored in a directory named for that person. This organization enables easy retrieval of all images for a given individual, which is necessary for generating positive pairs.

A dataset configuration file lists the names of all person directories, the number of images per person, and metadata such as capture date. This configuration is read by the training script to generate pairs.

The organized structure also enables incremental updates to the dataset. New images can be added to existing person directories, and the configuration file can be updated without regenerating the entire structure.

---

## 5.3 Data Preprocessing Pipeline

### 5.3.1 Preprocessing Overview

The preprocessing pipeline transforms raw input images into the format expected by the neural network. This pipeline is applied consistently to all images, whether for training, validation, or inference.

The pipeline consists of several stages: face detection, face extraction, resizing, color conversion, and pixel normalization. Each stage transforms the image in a specific way, and the stages are applied in sequence.

Preprocessing is implemented as a function that accepts an image path and returns a preprocessed numpy array. This function is called by both the training pipeline and the inference pipeline, ensuring consistency.

### 5.3.2 Face Detection Stage

The face detection stage locates faces in the input image using Haar cascade classification. This stage is crucial because subsequent stages operate on the detected face region.

The implementation uses OpenCV's Haar cascade classifier for frontal faces. The classifier is loaded once at module initialization and reused for all detections to avoid repeated file I/O.

The detectMultiScale function performs detection at multiple scales. The scaleFactor of 1.3 means the image is repeatedly scaled down by thirty percent between detection passes. The minNeighbors of 5 specifies that at least five neighboring detections are required to accept a face.

If no face is detected, the function returns None to indicate failure. In training, failed detections are logged and the image is excluded from the dataset. In inference, failed detections result in an error response to the user.

### 5.3.3 Face Extraction Stage

The face extraction stage crops the detected face region from the original image. This stage removes irrelevant background content and focuses on the face itself.

The detected face is defined by bounding box coordinates. The implementation adds padding around the bounding box to ensure the complete face is captured, including forehead and chin that may extend beyond the tight bounding box.

The padding amount is calculated as twenty percent of the box dimensions. This percentage was found through experimentation to work well for typical webcam captures. Too little padding cuts off face edges, while too much includes excessive background.

The padded coordinates are clipped to the image boundaries to handle cases where the detected face is near the image edge. The cropping operation uses numpy array slicing to extract the face region.

### 5.3.4 Resizing Stage

The resizing stage transforms the extracted face to the fixed dimensions expected by the neural network. All input images must be exactly ninety-six by ninety-six pixels.

The resize operation uses OpenCV's resize function with bilinear interpolation. Bilinear interpolation considers the four nearest source pixels to estimate each destination pixel, providing good quality for both upscaling and downscaling.

The aspect ratio is not preserved during resizing. The detected face may have been wider than tall or taller than wide, but the resize forces it to square dimensions. This may cause slight distortion, but the network learns to handle this variation.

### 5.3.5 Color Conversion Stage

The color conversion stage ensures the image channels are in the expected order. OpenCV reads images in BGR format, but the neural network expects RGB format.

The conversion is performed using OpenCV's cvtColor function with the COLOR_BGR2RGB flag. This rearranges the channel order without modifying the pixel values themselves.

Some implementations convert to grayscale, but this project uses color images throughout. Color provides additional discriminative information for face recognition, particularly for skin tone and eye color features.

### 5.3.6 Normalization Stage

The normalization stage scales pixel values to the range expected by the neural network. Raw pixel values range from zero to two hundred fifty-five, but networks typically train better with normalized inputs.

The normalization formula scales to the range negative one to positive one: normalized equals raw divided by one hundred twenty-seven point five minus one. This centers the data around zero with a range of two.

The normalization is applied after converting to floating-point dtype. The uint8 input is first converted to float32, then divided and subtracted to produce the normalized values.

The normalized image is returned as a numpy array with shape ninety-six by ninety-six by three and dtype float32. This array is ready for input to the neural network.

---

## 5.4 Network Architecture Implementation

### 5.4.1 Embedding Network Implementation

The embedding network transforms input images into fixed-dimensional embedding vectors. This implementation uses MobileNetV2 as the feature extraction backbone.

The embedding network is implemented as a Keras Model that takes an input image and produces an embedding vector. The model consists of the MobileNetV2 backbone followed by custom dense layers.

The MobileNetV2 backbone is loaded with pretrained ImageNet weights using the MobileNetV2 function from TensorFlow Keras applications. The include_top parameter is set to False to exclude the classification head. The pooling parameter is set to avg for global average pooling.

The backbone is frozen by setting trainable to False on the model. This preserves the pretrained weights and prevents them from being modified during training. Only the custom dense layers will be trained.

After the backbone, a Dense layer with five hundred twelve units and ReLU activation processes the backbone output. A BatchNormalization layer normalizes the activations, and a Dropout layer with rate point five provides regularization.

A second Dense layer with two hundred fifty-six units and ReLU activation further processes the features. Another BatchNormalization and Dropout with rate point three follow.

The final Dense layer has two hundred fifty-six units with no activation (linear output). A custom L2Normalization layer normalizes the output to unit length.

### 5.4.2 Comparison Network Implementation

The comparison network takes pairs of embeddings and produces a similarity score. This implementation concatenates L1 distance and cosine similarity features before passing through dense layers.

The comparison is implemented as part of the full Siamese model. The model takes two inputs (the two images to compare) and produces one output (the similarity score).

The embedding network is created once and shared between the two branches. The shared network ensures that both inputs are processed by identical transformations.

The L1 distance is computed by a Lambda layer that subtracts one embedding from the other and applies absolute value. The cosine similarity is computed by another Lambda layer that normalizes embeddings and computes their dot product.

The L1 distance and cosine similarity features are concatenated along the feature axis. The concatenation produces a vector with two hundred fifty-seven elements (two hundred fifty-six from L1 distance plus one from cosine similarity).

The concatenated features pass through three Dense layers with five hundred twelve, two hundred fifty-six, and one hundred twenty-eight units respectively. Each Dense layer is followed by BatchNormalization and Dropout.

The final Dense layer has one unit with sigmoid activation, producing a probability between zero and one. This represents the model's confidence that the two input images show the same person.

### 5.4.3 Custom Layer Definitions

The Siamese model uses several custom layers that must be defined before the model can be loaded. These layers encapsulate specific operations used in the architecture.

The L2Normalization layer extends the Keras Layer class and implements the l2_normalize operation. It normalizes inputs along the specified axis to have unit Euclidean norm. The get_config method enables serialization.

The L1Dist layer computes the element-wise absolute difference between two inputs. It receives two tensors of the same shape and returns their absolute difference.

The CosineSimilarity layer computes the cosine similarity between two input vectors. It normalizes both inputs to unit length and computes their dot product.

These custom layers are registered with their names when loading the model. The registration enables the model to be saved and loaded without losing layer definitions.

### 5.4.4 Model Compilation

Before training, the model must be compiled with an optimizer, loss function, and metrics. The compilation configures the training process.

The optimizer is Adam with an initial learning rate of point zero zero zero one. Adam combines momentum and adaptive learning rates for efficient optimization. The learning rate may be adjusted during training through a callback.

The loss function is binary cross-entropy, appropriate for the binary classification task of same versus different. The loss measures the difference between predicted probabilities and true labels.

The metrics track accuracy during training and evaluation. Accuracy measures the proportion of correct predictions and provides an intuitive sense of model performance.

---

## 5.5 Training Configuration

### 5.5.1 Training Hyperparameters

Training hyperparameters control the learning process and significantly affect the resulting model quality. The hyperparameters were selected based on experimentation and established best practices.

The learning rate of point zero zero zero one is the primary control over how quickly the model adapts to the training data. Higher rates learn faster but may overshoot optimal weights. Lower rates learn more slowly but may find better solutions.

The batch size of sixty-four determines how many training pairs are processed before weight updates. Larger batches provide more accurate gradient estimates but require more memory. The batch size of sixty-four provides good balance.

The maximum epochs of twenty limits how long training can run. Early stopping typically terminates training before reaching this limit when validation performance plateaus. Twenty epochs is sufficient for the model to converge.

### 5.5.2 Callback Configuration

Callbacks are functions that are called at various points during training. They enable monitoring, checkpointing, and adaptive behavior.

The ModelCheckpoint callback saves the model after each epoch if validation loss improved. The save_best_only parameter ensures only the best model is kept. The model is saved in Keras format with the .keras extension.

The EarlyStopping callback monitors validation loss and stops training if no improvement is seen for five epochs. The restore_best_weights parameter ensures the model is reverted to the best weights rather than the weights from the last epoch.

The ReduceLROnPlateau callback reduces the learning rate when validation loss plateaus. The factor of point five halves the learning rate, and the patience of three epochs waits before reducing.

### 5.5.3 Data Augmentation Configuration

Data augmentation is configured to apply random transformations that increase effective training set diversity. The augmentations simulate real-world variations.

Horizontal flipping is enabled with fifty percent probability. This creates variations that simulate looking at the face from left or right. The mirroring is valid because left and right are arbitrary distinctions.

Brightness adjustment multiplies pixel values by a random factor between point seven and one point three. This simulates different lighting conditions from dim to bright.

Contrast adjustment scales pixel values relative to the mean with factors between point seven and one point three. This simulates different camera contrast settings.

Rotation applies random rotations up to twenty degrees in either direction. This simulates slight head tilts that commonly occur in webcam captures.

Gaussian noise with standard deviation point zero two adds random variation to pixel values. This simulates sensor noise and minor image quality variations.

Translation shifts the image by up to eight pixels horizontally and vertically. This simulates positioning variations in webcam placement.

---

## 5.6 Training Process

### 5.6.1 Training Initialization

Training begins with initialization steps that set up the environment and load data.

The training script first imports all required libraries including TensorFlow, Keras, OpenCV, NumPy, and the custom model components. TensorFlow logging is set to suppress informational messages.

The model architecture is instantiated by calling the model creation function. This builds the complete Siamese network with embedding and comparison components.

The model is compiled with the configured optimizer, loss, and metrics. After compilation, the model is ready for training but has not yet seen any data.

The training data is loaded by the dataset generator. The generator reads images from the organized directory structure, applies preprocessing, and generates pairs on-the-fly.

### 5.6.2 Training Loop

The training loop iterates over epochs, each epoch processing all training pairs once.

An epoch begins with the training data generator producing batches of pairs and labels. Each batch contains sixty-four pairs as configured by the batch size.

For each batch, the model computes predictions, calculates loss, and updates weights through backpropagation. The optimizer adjusts weights in the direction that reduces loss.

After processing all batches, the epoch ends with validation on the held-out validation set. Validation runs the model on validation pairs without weight updates to assess generalization.

Training metrics including loss and accuracy are recorded for each epoch. These metrics are written to a history file for later analysis and visualization.

### 5.6.3 Training Monitoring

During training, several metrics are monitored to assess progress and detect issues.

The training loss measures how well the model fits the training data. The loss should decrease over epochs, indicating the model is learning.

The validation loss measures how well the model generalizes to unseen data. If validation loss increases while training loss decreases, the model is overfitting.

The training and validation accuracy measure the proportion of correct predictions. Accuracy should increase over epochs and ideally be similar for training and validation.

The learning rate may change during training if the ReduceLROnPlateau callback triggers. Lower learning rates enable fine-tuning in later epochs.

### 5.6.4 Checkpoint Management

The training process saves checkpoints that capture the model state at different points.

The ModelCheckpoint callback saves the model after each epoch if validation loss improved. This ensures the best model is preserved even if later epochs degrade performance.

Checkpoints are saved in HDF5 format with the .h5 extension. The file contains both the model architecture and the trained weights.

The training script records which epoch produced the best checkpoint. After training completes, this best checkpoint is copied to the final model file.

---

## 5.7 Training Results

### 5.7.1 Training Progress

The training process produced a model with excellent performance on both training and validation data. The following summarizes the key metrics observed during training.

The training loss started at approximately point seven nine in the first epoch and decreased rapidly through the first five epochs. By epoch ten, the training loss had decreased to approximately point zero three. The final training loss reached approximately point zero zero eight nine.

The validation loss followed a similar trajectory, starting high and decreasing to approximately point zero zero zero zero four. The validation loss was consistently lower than training loss, which is unusual but indicates the model generalized well to the validation pairs.

The training accuracy reached approximately ninety-nine point seven seven percent by the final epoch. This indicates the model correctly classified nearly all training pairs.

The validation accuracy reached one hundred percent, meaning all validation pairs were correctly classified. This exceptional result should be interpreted with caution given the limited validation set size.

### 5.7.2 Final Model Characteristics

The trained model has specific characteristics that reflect the training process and architecture choices.

The model size is approximately twelve megabytes when saved in HDF5 format. This compact size enables fast loading and moderate memory requirements during inference.

The embedding dimension of two hundred fifty-six means each face is represented by a vector of two hundred fifty-six numbers. This compact representation enables efficient storage and fast comparison.

The parameter count of approximately three million one hundred fifty-eight thousand includes both trainable and non-trainable parameters. Most parameters (approximately two million seven hundred thousand) are in the frozen MobileNetV2 backbone.

### 5.7.3 Training Curves

The training history is recorded and can be visualized to understand the learning process.

The accuracy curve shows rapid improvement in the first few epochs, reaching high accuracy by epoch five. After that, accuracy continued to improve gradually until reaching near-perfect levels.

The loss curve shows corresponding rapid decrease in the early epochs, followed by gradual decrease as the model fine-tunes. The logarithmic scale is often used for loss to better visualize small values.

The learning rate schedule shows step decreases when the ReduceLROnPlateau callback triggered. These decreases enabled finer optimization in later epochs.

---

## 5.8 Model Evaluation

### 5.8.1 Evaluation Methodology

The trained model was evaluated on held-out test data to characterize its real-world performance. The evaluation methodology ensures unbiased performance estimates.

The test set consists of three thousand pairs that were not seen during training or validation. These pairs represent the true distribution of same and different person comparisons that the system will encounter.

For each test pair, the model predicts a similarity score. The prediction is compared to the true label to determine if the prediction was correct. Correct predictions are true positives or true negatives; incorrect predictions are false positives or false negatives.

Multiple metrics are computed from the predictions including accuracy, precision, recall, and F1 score. Each metric provides different insight into model performance.

### 5.8.2 Confusion Matrix Analysis

The confusion matrix provides a detailed breakdown of prediction outcomes. For the verification task with a threshold of point five, the confusion matrix shows how many pairs fell into each category.

True positives are positive pairs (same person) correctly predicted as positive. These represent correct matches where the system identified the same person.

True negatives are negative pairs (different people) correctly predicted as negative. These represent correct rejections where the system correctly identified different people.

False positives are negative pairs incorrectly predicted as positive. These represent security concerns where different people were incorrectly matched.

False negatives are positive pairs incorrectly predicted as negative. These represent user inconvenience where the same person was incorrectly rejected.

### 5.8.3 Performance Metrics

The model achieved excellent performance on all evaluated metrics.

Accuracy measures the proportion of all predictions that were correct. The model achieved approximately one hundred percent accuracy on the test set, correctly classifying all three thousand test pairs.

Precision measures the proportion of positive predictions that were correct. With perfect accuracy, precision is also approximately one hundred percent.

Recall measures the proportion of actual positives that were correctly predicted. With perfect accuracy, recall is also approximately one hundred percent.

F1 score is the harmonic mean of precision and recall. With perfect precision and recall, the F1 score is approximately one point zero.

### 5.8.4 Limitations and Caveats

While the evaluation results are excellent, several limitations should be acknowledged.

The test set size of three thousand pairs, while substantial, may not fully represent the diversity of real-world inputs. Larger test sets would provide more reliable estimates.

The test set was generated from the same pool of images as the training set. Real-world inputs may have characteristics not represented in the training data.

The controlled capture conditions may not reflect the full range of real-world variations in lighting, pose, expression, and image quality.

The perfect accuracy result should be treated with some skepticism. It may indicate that the test set is not sufficiently challenging, rather than that the model has achieved perfect performance.

---

## 5.9 Chapter Summary

This chapter has documented the complete process of developing and training the Siamese neural network model for face verification.

The data collection section described the sources, statistics, and specifications of the training dataset. The organized structure enables efficient pair generation and incremental updates.

The preprocessing pipeline section explained each stage of image transformation from raw input to network-ready array. The pipeline is consistent between training and inference.

The network architecture section detailed the embedding and comparison components, including the MobileNetV2 backbone and custom layers. The architecture balances accuracy and efficiency.

The training configuration section specified the hyperparameters, callbacks, and augmentation that control the learning process. These settings were selected based on experimentation.

The training process section described the initialization, loop, monitoring, and checkpoint management that produce the final model. The training achieved excellent results.

The evaluation section presented the performance metrics and acknowledged the limitations of the evaluation methodology.

With the model documented, the report proceeds to Chapter 6, which covers the implementation of the complete system including backend, frontend, and desktop applications.

---

**End of Chapter 5**
