# CHAPTER 3: SYSTEM METHODOLOGY

---

## 3.1 Introduction to System Methodology

### 3.1.1 Overview of the Methodology

This chapter provides an in-depth explanation of how the face recognition system works, detailing each component of the processing pipeline from the moment a user captures an image to when the system returns an identity verification result. The methodology described here represents the complete end-to-end process that transforms raw pixel data into meaningful recognition decisions.

The face recognition system developed in this project employs a Siamese neural network architecture with MobileNetV2 as the feature extraction backbone. This combination was chosen to balance recognition accuracy with computational efficiency, enabling the system to run on standard consumer hardware without requiring specialized graphics processing units. The methodology prioritizes practical implementation details that can be directly understood and reproduced.

Understanding the system methodology is essential for several reasons. For developers extending or modifying the system, the methodology explains why each component is designed as it is and how changes might affect overall performance. For users and administrators, the methodology provides insight into how the system processes their data, which is important for understanding system behavior and limitations. For evaluators and reviewers, the methodology demonstrates the technical depth of the project and the student's understanding of the underlying concepts.

The methodology is organized into logical sections covering each major aspect of the system. Each section explains not just what happens but why it happens that way, including the trade-offs considered and the alternatives evaluated. This explanatory approach provides context that goes beyond simple description.

### 3.1.2 System Architecture Overview

The overall system architecture consists of three primary layers: the presentation layer, the processing layer, and the data layer. Each layer has distinct responsibilities and communicates with other layers through well-defined interfaces.

The presentation layer encompasses all user-facing components through which users interact with the system. This includes the React-based web application accessible from any modern browser, the Kivy-based desktop application for standalone operation, and the API documentation interface provided by FastAPI. The presentation layer handles user input, displays results, and manages the overall user experience.

The processing layer contains the core intelligence of the system, including the trained Siamese neural network, image preprocessing routines, and the business logic for handling enrollment and verification requests. This layer receives data from the presentation layer, processes it according to established procedures, and returns results. The processing layer is stateless and can be scaled horizontally by adding more server instances.

The data layer manages persistent storage of models, enrolled person data, and system configuration. The trained neural network model is stored as an HDF5 file containing model architecture and weights. Person enrollment data is stored as image files organized in directories, with metadata in JSON format. Configuration settings are stored separately to enable runtime adjustment without code changes.

The three-layer separation enables independent development and testing of each component. Changes to the user interface do not require modifications to the processing logic, and vice versa. This modularity simplifies maintenance and enables technology upgrades without wholesale system redesign.

### 3.1.3 High-Level Processing Flow

When a user submits an image for recognition, the system performs a sequence of operations that transform the input into a verification decision. Understanding this flow is fundamental to understanding the entire system.

The process begins when the presentation layer receives the image file from the user. This may be through webcam capture, file upload, or API request. The presentation layer performs basic validation such as checking that a file was actually received and that it has an acceptable format.

The presentation layer sends the image to the processing layer through an HTTP request. For the web application, this is an AJAX request to the FastAPI backend. For the desktop application, this is a local HTTP request to the same backend service. The API endpoint receives the request and initiates processing.

The processing layer first saves the received image to a temporary file. This enables the image processing functions to work with the file system rather than in-memory data, which simplifies error handling and enables streaming for large files. After saving, the processing pipeline begins.

The first pipeline stage is face detection, which locates any faces in the image. If no face is detected, the pipeline returns an error indicating that the image does not contain a detectable face. If multiple faces are detected, the system may either reject the image as ambiguous or use only the largest detected face, depending on configuration.

The second pipeline stage is face preprocessing, which extracts and normalizes the detected face. The detected face region is cropped from the original image with appropriate padding. The cropped face is resized to the dimensions expected by the neural network, which is ninety-six by ninety-six pixels in this project. The pixel values are normalized to the range negative one to positive one.

The third pipeline stage is embedding extraction, which runs the preprocessed face through the neural network to obtain a compact representation. The Siamese network has two identical branches that share weights; for single-image operations, only one branch is used. The network output is a two hundred fifty-six-dimensional vector that encodes the facial identity information.

The fourth pipeline stage is comparison, which compares the extracted embedding against stored embeddings for enrolled persons. For each enrolled person, the system computes the cosine similarity between the query embedding and the stored embedding or average of stored embeddings for that person. The highest similarity score determines the best match.

The fifth pipeline stage is decision, which compares the best match similarity against the configured threshold. If the similarity exceeds the threshold, the person is identified as a match. If the similarity is below the threshold, the person is identified as unknown. The system returns the identification result along with the similarity scores.

### 3.1.4 Enrollment Flow

The enrollment process adds new persons to the system database, enabling subsequent recognition. Enrollment is more complex than recognition because it must handle multiple images per person and create persistent storage.

Enrollment begins when a user provides their name and one or more facial images. The name is used to create a unique identifier for the person. The images are captured through webcam or uploaded from file. The system recommends capturing at least three images from different angles to improve recognition accuracy.

For each submitted image, the system runs the full recognition pipeline through the embedding extraction stage. If face detection fails on an image, the system returns an error indicating which image could not be processed. The user must provide a replacement image before enrollment can proceed.

After successfully extracting embeddings from all images, the system creates a directory for the person using a slugified version of their name as the directory name. The slugification process converts spaces to underscores, removes special characters, and converts to lowercase to create a valid directory name. For example, "John Doe" becomes "john_doe".

The images are saved to the person's directory with sequential numeric names. The embeddings are stored in a metadata file alongside the images. The metadata includes the person's display name, enrollment timestamp, image count, and individual embeddings. This metadata enables the system to load person information without reprocessing images.

After enrollment completes, the system returns a success message indicating how many images were enrolled for the new person. The person immediately becomes available for recognition. No model retraining is required because the Siamese network architecture does not require knowing all identities during training.

---

## 3.2 Face Detection Implementation

### 3.2.1 Face Detection Fundamentals

Face detection is the process of automatically locating faces within digital images. It is a prerequisite for face recognition because the recognition algorithm must know where in the image to focus its attention. Without accurate face detection, even the most sophisticated recognition algorithm will fail because it may be analyzing the wrong region of the image.

Modern face detection algorithms can handle substantial variation in pose, lighting, expression, and occlusion. They output bounding boxes indicating the location and size of detected faces, along with confidence scores indicating how certain the detector is of each detection. Advanced detectors also output facial landmarks indicating the positions of eyes, nose, mouth, and face contour.

The importance of face detection extends beyond simply locating faces. The detected face region enables cropping that removes irrelevant background content. The bounding box dimensions inform appropriate padding for the crop. The confidence score enables filtering of uncertain detections. The landmarks enable geometric normalization for pose compensation.

This project uses the Haar Cascade classifier implemented in OpenCV for face detection. While more modern detectors like MTCNN or RetinaFace offer improved accuracy, particularly for non-frontal poses, the Haar Cascade provides acceptable performance with minimal complexity and no additional model files.

### 3.2.2 Haar Cascade Classifier

The Haar Cascade classifier is based on the Viola-Jones object detection framework published in 2001. Despite its age, it remains popular for its speed, simplicity, and reasonable accuracy on frontal faces under typical conditions. Understanding how it works provides insight into object detection principles that apply more broadly.

The classifier uses a cascade structure consisting of multiple stages. Each stage contains a set of features and a threshold. An image region is evaluated by each stage in sequence. If a stage determines the region cannot contain a face, it is immediately rejected without evaluating subsequent stages. Only regions that pass all stages are classified as containing faces.

The features used in Haar cascades are based on Haar wavelets, which are simple rectangular patterns. A feature computes the difference between the sum of pixel values in white and black regions. For example, one feature type computes the difference between the eye region and the bridge of the nose. These simple features capture some facial structure without requiring complex shape models.

The cascade is trained using a procedure called boosting. Early stages are trained to reject most non-face regions with minimal computation, while later stages perform more detailed analysis on promising regions. This design enables real-time performance because most image regions are rejected early with simple tests.

OpenCV includes several pretrained Haar cascade classifiers for different face orientations and body parts. The frontal face default classifier is trained on thousands of positive and negative examples to recognize frontal faces. Alternative classifiers exist for profile faces, eyes, and other features.

### 3.2.3 Implementation Details

The face detection implementation in this project wraps the OpenCV Haar cascade functionality in a function that accepts an image path or numpy array and returns the detected face regions. This wrapper provides consistent interface and handles the conversion between different image formats.

The implementation begins by loading the Haar cascade XML file using OpenCV's CascadeClassifier class. OpenCV includes several pretrained cascades; this project uses haarcascade_frontalface_default.xml for frontal face detection. The file is loaded once at application startup and reused for all detections.

When detecting faces in an image, the implementation first converts the image to grayscale. Color information is not useful for face detection because the Haar features operate on intensity patterns rather than colors. Grayscale conversion reduces the image to a single channel, simplifying processing.

The detectMultiScale function performs the actual detection. The scaleFactor parameter controls how much the image is resized between detection passes. A value of 1.3 means the image is repeatedly scaled down by thirty percent until reaching a minimum size. This multi-scale approach enables detection of faces at different sizes. The minNeighbors parameter specifies how many neighboring detections are required to accept a face. Higher values reduce false positives but may miss some faces. A value of five provides reasonable trade-off.

The function returns a list of rectangles, each defined by the x and y coordinates of the top-left corner, the width, and the height. For images with multiple detected faces, this project currently uses only the largest face for simplicity. In a production system, multiple faces might be handled by processing each separately or by alerting the user to the ambiguous input.

### 3.2.4 Handling Detection Failures

Robust face detection must handle various failure modes gracefully. Common failures include no face detected, multiple faces detected, face too small, or face partially outside the image bounds. The implementation addresses each failure mode appropriately.

When no face is detected, the system cannot proceed with recognition or enrollment. The user interface should display an informative message indicating that no face was found and suggest adjusting lighting, facing the camera directly, or moving closer. For programmatic access, the API returns an error code indicating detection failure.

When multiple faces are detected, the system faces ambiguity. For enrollment, using a photo with multiple people could inadvertently enroll all of them. For recognition, matching against multiple potential subjects creates confusion. The current implementation uses only the largest detected face, which is a simple heuristic that works in many cases but is not ideal.

When a detected face is very small, the resolution may be insufficient for accurate recognition. The implementation enforces a minimum face size relative to the image dimensions. Faces below this threshold are rejected with an appropriate error message suggesting the subject move closer to the camera.

When a detected face extends beyond image boundaries, the face region is clipped to the valid bounds. This truncation may remove part of the face, potentially reducing recognition accuracy. The padding calculation attempts to ensure sufficient margin around detected faces while staying within image bounds.

---

## 3.3 Image Preprocessing Pipeline

### 3.3.1 Purpose of Preprocessing

Image preprocessing transforms raw input images into a standardized format suitable for the neural network. Preprocessing ensures that the network receives consistent inputs regardless of variations in the original images, which is essential for reliable performance.

The neural network was trained with specific expectations about its inputs. The input images must be exactly ninety-six pixels wide and ninety-six pixels tall. The pixel values must be normalized to the range negative one to positive one. Any deviation from these expectations degrades network performance.

Beyond meeting the network's expectations, preprocessing addresses real-world image variations. Different cameras produce images with different color characteristics. Different lighting conditions produce different exposure levels. Different distances from the camera produce different face sizes. Preprocessing normalizes these variations to consistent representations.

Preprocessing also enables data augmentation by providing a standardized starting point. Augmentation operations like rotation and scaling are simpler when applied to normalized images of fixed size. The combination of preprocessing and augmentation creates an effective training pipeline.

### 3.3.2 Face Extraction and Cropping

After face detection provides bounding boxes for faces in the image, the system extracts and crops the face regions. This step removes irrelevant background content and focuses the subsequent processing on the face itself.

The extraction process begins with the detected bounding box coordinates. The box may not include the complete face, particularly the forehead and chin. The implementation adds padding around the detected box to ensure the complete face is included. The padding is calculated as a percentage of the box dimensions, typically twenty percent.

After padding, the coordinates are clipped to the image boundaries. Any portion of the padded box that extends beyond the image is removed. This may result in slightly less padding on some sides, but ensures the crop region is entirely within the image.

The cropped region is extracted using numpy array slicing. The image array is indexed to select only the rows and columns corresponding to the crop region. This produces a smaller image containing just the face and surrounding context.

The cropping step significantly affects recognition accuracy. Too little padding may cut off parts of the face, removing discriminative features. Too much padding includes too much background, diluting the face information with irrelevant content. The twenty percent padding was found through experimentation to work well for typical webcam captures.

### 3.3.3 Resizing to Network Input Size

The extracted face image must be resized to match the network's expected input dimensions. For this project, the network expects ninety-six by ninety-six pixel RGB images. The resizing operation transforms any input face to this size.

OpenCV's resize function performs the resizing using interpolation. When scaling down, the function must estimate pixel values at the new lower-resolution grid from the original higher-resolution grid. When scaling up, the function must estimate the new higher-resolution grid from the original lower-resolution grid.

The implementation uses bilinear interpolation, which considers the four nearest original pixels to estimate each new pixel. Bilinear interpolation provides a good balance between quality and computation. More advanced interpolation methods like bicubic or Lanczos produce slightly better quality at higher computational cost.

When resizing, the aspect ratio of the face may change if the detected face was not square. The implementation resizes to exact dimensions without preserving aspect ratio, which may cause some horizontal or vertical stretching. This distortion is typically minor and does not significantly affect recognition accuracy.

For training, the input size of ninety-six pixels was chosen to balance detail preservation with computational efficiency. Larger inputs capture more detail but require more computation. Smaller inputs are faster but may lose discriminative features. Ninety-six pixels provides good trade-off for this application.

### 3.3.4 Color Space Conversion

OpenCV reads images in BGR format by default, with the blue channel in the first position, green in the second, and red in the third. The neural network was trained expecting RGB format with the red channel first. Color space conversion ensures the network receives channels in the expected order.

The conversion from BGR to RGB is performed using OpenCV's cvtColor function with the COLOR_BGR2RGB flag. This function rearranges the channel order without modifying the pixel values themselves. The red, green, and blue intensities for each pixel remain the same; only their storage positions change.

Some implementations convert to grayscale for simplified processing. This project uses color images throughout because color provides additional discriminative information for face recognition. Certain facial features like skin tone variation and lip color are lost in grayscale conversion.

### 3.3.5 Pixel Value Normalization

Neural networks train more effectively when inputs are normalized to a bounded range. Raw pixel values typically span from zero (black) to two hundred fifty-five (white). The network expects values in the range negative one to positive one.

The normalization formula used in this project is: normalized equals raw divided by one hundred twenty-seven point five minus one. This divides the zero-to-two-hundred-fifty-five range by one hundred twenty-seven point five to get the range zero-to-two, then subtracts one to shift to the range negative one-to-positive-one.

This normalization scheme centers the data around zero, which helps with gradient-based optimization during training. Values near zero are neither bright nor dark, which is typically the case for well-exposed faces. Extreme values like pure black or pure white are less common in real faces.

The normalization is applied after resizing and color conversion. The input image is first resized to ninety-six by ninety-six pixels, then converted from BGR to RGB, then normalized to the negative-one-to-positive-one range. These operations are performed in sequence to produce the final preprocessed image.

### 3.3.6 Preprocessing Code Implementation

The preprocessing pipeline is implemented as a function that accepts an image file path and returns a preprocessed numpy array ready for the neural network. The function encapsulates all preprocessing steps and provides a single entry point for the rest of the system.

The function begins by reading the image from the file path using OpenCV's imread function. If imread returns None, the file could not be read, and the function returns None to indicate failure. This may occur if the file does not exist, is corrupted, or is not a valid image format.

After reading, the function performs face detection using the Haar cascade. If no face is detected, the function returns None with an appropriate error indicator. If multiple faces are detected, the function currently uses the largest face by area (width times height).

The face extraction and all subsequent steps operate on the detected face region. The padding calculation adds twenty percent margin to each side of the detected bounding box. The coordinates are clipped to image bounds, and the crop is extracted.

The resize operation uses INTER_LINEAR interpolation, which provides good quality for both upscaling and downscaling. The result is a ninety-six by ninety-six by three numpy array with uint8 dtype (values zero to two hundred fifty-five).

Color conversion produces a ninety-six by ninety-six by three array in RGB order. The normalization converts the dtype to float32 and scales values to negative one to positive one. The final result is a ninety-six by ninety-six by three float32 array ready for the neural network.

---

## 3.4 Siamese Neural Network Architecture

### 3.4.1 Architecture Design Principles

The Siamese neural network architecture was designed to learn an effective similarity metric for face verification. The design prioritizes several key principles that guide the component choices and configurations.

The first principle is parameter efficiency. The network should use as few trainable parameters as possible while maintaining accuracy. Fewer parameters reduce memory requirements, speed up inference, and decrease the risk of overfitting. MobileNetV2 was chosen as the backbone precisely because it achieves excellent results with fewer parameters than alternatives.

The second principle is representation quality. The embeddings produced by the network should capture identity-relevant information while being invariant to irrelevant variations. The embedding space should place images of the same person close together and images of different people far apart.

The third principle is computational efficiency. The network should process images quickly on CPU hardware without specialized acceleration. The MobileNetV2 backbone with ninety-six-pixel inputs achieves inference times well under one second on typical hardware.

The fourth principle is training stability. The architecture should train reliably without requiring extensive hyperparameter tuning or careful learning rate scheduling. Batch normalization and careful initialization help ensure stable training.

### 3.4.2 Embedding Network Architecture

The embedding network is the core component that transforms input images into compact vector representations. This network is shared between the two branches of the Siamese architecture, hence the name "Siamese" referring to the identical twin networks.

The embedding network begins with MobileNetV2 as a frozen feature extraction backbone. MobileNetV2 was initialized with ImageNet weights and kept frozen during training. The final pooling layer of MobileNetV2 outputs a one thousand two hundred eighty-dimensional feature vector for each input image.

After the MobileNetV2 backbone, two fully connected layers process the feature vector. The first dense layer has five hundred twelve units with ReLU activation, followed by batch normalization and dropout with rate point five. Batch normalization stabilizes training by normalizing activations, and dropout prevents overfitting by randomly zeroing some activations.

The second dense layer has two hundred fifty-six units with ReLU activation, followed by batch normalization and dropout with rate point three. This layer further reduces the dimensionality and extracts higher-level features.

The final layer produces the two hundred fifty-six-dimensional embedding with no activation (linear output). A custom L2 normalization layer ensures the embedding vectors have unit length, which simplifies the similarity comparison.

The complete embedding network has approximately three million parameters, with approximately two hundred fifty-six thousand trainable parameters in the two dense layers. The frozen MobileNetV2 backbone contributes approximately two point seven million non-trainable parameters.

### 3.4.3 Comparison Network Architecture

The comparison network takes the embeddings from two input images and produces a similarity score. This component implements the verification decision logic that determines whether two images show the same person.

The comparison begins by computing two distance features from the embeddings. The L1 distance computes the absolute element-wise difference between the two embeddings. The cosine similarity computes the normalized dot product. These two features capture different aspects of the relationship between embeddings.

The L1 distance is computed by a custom layer that subtracts one embedding from the other and takes the absolute value. This produces a two hundred fifty-six-dimensional vector where each element represents the absolute difference at that dimension.

The cosine similarity is computed by a custom layer that normalizes both embeddings to unit length and computes their dot product. This produces a scalar value between negative one and one, where higher values indicate greater similarity.

The L1 distance and cosine similarity features are concatenated to form a two hundred fifty-seven-dimensional vector. This concatenation combines the complementary information from both distance measures.

The concatenated features pass through three fully connected layers. The first layer has one hundred twenty-eight units with ReLU activation and dropout at point four. The second layer has sixty-four units with ReLU activation and dropout at point three. The third layer has thirty-two units with ReLU activation.

The final output layer has a single unit with sigmoid activation, producing a value between zero and one. This value represents the probability that the two input images show the same person. Values close to one indicate high confidence of match, while values close to zero indicate high confidence of non-match.

### 3.4.4 Custom Layer Implementations

The Siamese network uses several custom layer implementations beyond the standard Keras layers. These custom layers encapsulate specific operations that are reused in multiple places within the architecture.

The L2 normalization layer normalizes its input to have unit Euclidean norm along the specified axis. For face embeddings, normalization ensures that similarity depends only on the direction of the vectors, not their magnitude. The layer is implemented as a Keras layer that calls tf.math.l2_normalize.

The L1 distance layer computes the element-wise absolute difference between two inputs. This is a Siamese-specific operation that cannot be achieved with standard layers. The layer receives two tensors of equal shape and returns their absolute difference.

The cosine similarity layer computes the cosine of the angle between two input vectors. The implementation normalizes both inputs to unit length using L2 normalization, then computes their dot product. This produces the cosine similarity value directly.

All custom layers implement the get_config method to support model serialization and deserialization. This enables saving and loading the complete model including custom layer definitions. Without proper serialization support, the model could not be saved to disk and reloaded later.

### 3.4.5 Network Summary and Parameters

The complete Siamese network architecture can be summarized as follows. The input accepts pairs of ninety-six by ninety-six by three images. The embedding network processes each image through MobileNetV2 (frozen), followed by Dense(1280, ReLU), BatchNorm, Dropout(0.5), Dense(512, ReLU), BatchNorm, Dropout(0.3), and Dense(256, linear) with L2 normalization.

The comparison network takes the two embeddings and computes L1 distance and cosine similarity. These features are concatenated and processed through Dense(128, ReLU), BatchNorm, Dropout(0.4), Dense(64, ReLU), BatchNorm, Dropout(0.3), Dense(32, ReLU), and Dense(1, sigmoid) to produce the final similarity score.

The total parameter count is approximately three million one hundred fifty-eight thousand parameters, of which approximately eight hundred ninety-eight thousand are trainable. The non-trainable parameters come from the frozen MobileNetV2 backbone. The model size is approximately twelve megabytes when stored in HDF5 format.

This parameter count is quite modest compared to state-of-the-art face recognition models, which often have tens or hundreds of millions of parameters. The efficiency comes from using MobileNetV2 as the backbone rather than larger architectures like ResNet or VGG.

---

## 3.5 Similarity Metrics

### 3.5.1 Understanding Similarity Metrics

Similarity metrics quantify how alike two objects are based on their representations. In face recognition, similarity is measured between embedding vectors, with higher similarity indicating higher likelihood that the embeddings come from images of the same person.

The choice of similarity metric affects the geometry of the embedding space and consequently affects recognition performance. Different metrics have different mathematical properties and may suit different types of embeddings better. The implementation uses both L1 distance and cosine similarity, combining their complementary strengths.

Similarity metrics operate on the embedding vectors produced by the embedding network. The embeddings encode facial identity information in a two hundred fifty-six-dimensional space. Images of the same person should cluster together, while images of different people should be far apart.

The embedding space is not explicitly designed but emerges from the training process. The training objective encourages same-person pairs to have similar embeddings and different-person pairs to have dissimilar embeddings. The specific geometry of the resulting space depends on the training data and hyperparameters.

### 3.5.2 L1 Distance

L1 distance, also called Manhattan distance or city-block distance, measures the absolute difference between vectors along each dimension. For embeddings A and B with N dimensions, the L1 distance is the sum of absolute differences across all dimensions.

Mathematically, L1 distance equals the sum over dimensions i of the absolute value of A_i minus B_i. This is equivalent to taking the L1 norm of the difference vector. The L1 distance is always non-negative and equals zero only when the vectors are identical.

L1 distance captures the magnitude of difference between embeddings. Large L1 distance indicates substantial differences across multiple dimensions. Small L1 distance indicates embeddings that are close in every dimension.

The implementation computes L1 distance using a custom Keras layer that subtracts one embedding from the other and applies absolute value. The output is a two hundred fifty-six-dimensional vector of absolute differences, which is then processed by the comparison network.

### 3.5.3 Cosine Similarity

Cosine similarity measures the angle between vectors rather than their magnitudes. For embeddings A and B, cosine similarity equals the dot product of the normalized vectors. Values range from negative one (opposite directions) through zero (orthogonal) to positive one (identical directions).

Mathematically, cosine similarity equals the sum over dimensions i of A_i times B_i divided by the product of the L2 norms of A and B. When embeddings are unit-normalized, the denominator equals one, simplifying to just the dot product.

Cosine similarity is invariant to the scale of the vectors. Because embeddings are L2-normalized, only their direction matters, not their magnitude. This invariance is desirable for face recognition because image quality variations may affect embedding magnitude without affecting identity information.

The implementation computes cosine similarity using a custom Keras layer. The layer normalizes both inputs to unit length, computes their element-wise product, and sums across dimensions. The result is a scalar similarity value.

### 3.5.4 Combining Multiple Metrics

The implementation combines L1 distance and cosine similarity rather than using either alone. This combination leverages the complementary information captured by each metric.

L1 distance captures the raw magnitude of difference between embeddings. It is sensitive to the absolute position of embeddings in space. Two embeddings that differ by one unit in every dimension would have a large L1 distance even if they point in similar directions.

Cosine similarity captures the directional relationship between embeddings. It is insensitive to the magnitude of difference but sensitive to the orientation. Two embeddings that are scaled versions of each other would have maximum cosine similarity even with different L1 distances.

By concatenating both features, the comparison network has access to both types of information. The subsequent dense layers can learn optimal weights for combining these features, potentially learning that one metric is more reliable than the other in certain situations.

### 3.5.5 Threshold-Based Decision

The final step in verification is deciding whether the similarity score indicates a match or non-match. This is accomplished using a configurable threshold. Scores above the threshold are classified as matches, while scores below are classified as non-matches.

The threshold of point five provides reasonable default behavior, corresponding to greater than fifty percent probability of match according to the sigmoid output. Higher thresholds increase precision (fewer false positives) at the cost of lower recall (more false negatives). Lower thresholds increase recall at the cost of lower precision.

The appropriate threshold depends on the application requirements. Security-critical applications like access control might use higher thresholds to minimize false acceptance. Convenience-focused applications like photo tagging might use lower thresholds to minimize false rejection.

The system exposes the threshold as a configurable parameter. Users can adjust it through the settings interface. The default threshold of point five balances security and convenience for typical use cases.

---

## 3.6 Training Methodology

### 3.6.1 Training Data

The training data consists of pairs of face images labeled as positive (same person) or negative (different people). The Siamese network learns from these pairs to distinguish between same-person and different-person comparisons.

Positive pairs consist of two images of the same person. The training set includes multiple images of each enrolled subject, captured under varied conditions. These images form the basis for positive pair generation, with all combinations of two images from the same person creating positive pairs.

Negative pairs consist of two images of different people. The training set includes images from many different subjects. Randomly sampling images from different subjects creates negative pairs. The number of possible negative pairs grows with the square of the number of subjects, providing ample negative examples.

The total training set contains approximately fifteen thousand pairs, split eighty-twenty into training and validation sets. The training set has twelve thousand pairs, and the validation set has three thousand pairs. The split is stratified to maintain approximately equal numbers of positive and negative pairs in each set.

### 3.6.2 Data Augmentation

Data augmentation artificially expands the training set by applying random transformations to the original images. These transformations simulate real-world variations that the network should handle, improving generalization.

The augmentation pipeline applies multiple transformation types with specified probabilities. Horizontal flipping applies with fifty percent probability, mirroring the face left-to-right. This is valid for face recognition because left and right are arbitrary distinctions.

Brightness adjustment multiplies pixel values by a random factor between point seven and one point three. This simulates different lighting conditions, from dim to bright environments. The factor is drawn from a uniform distribution within this range.

Contrast adjustment scales pixel values relative to the mean intensity. Factors between point seven and one point three are applied. This simulates different camera contrast settings or lighting contrast ratios.

Rotation applies random rotations up to twenty degrees clockwise or counterclockwise. This simulates slight head tilts. The rotation is implemented using OpenCV's rotation matrix computation.

Gaussian noise with standard deviation point zero two adds random variation to pixel values. This simulates sensor noise and minor image quality variations.

Translation shifts the image by up to eight pixels horizontally and vertically. This simulates small positioning variations that commonly occur with webcam capture.

### 3.6.3 Training Configuration

The training configuration specifies the hyperparameters and procedures used to train the network. Careful configuration is essential for achieving good performance.

The optimizer is Adam with initial learning rate of point zero zero zero one. Adam combines momentum and adaptive learning rates for efficient optimization. The learning rate is reduced by a factor of point five if validation loss does not improve for three epochs.

The batch size is sixty-four, meaning the network processes sixty-four image pairs in each training step. Larger batch sizes provide more accurate gradient estimates but require more memory. Sixty-four provides good balance for the available hardware.

The loss function is binary cross-entropy, which measures the difference between predicted probabilities and true labels. For a verification problem, this compares the predicted match probability against the actual label of same or different.

Training runs for up to twenty epochs with early stopping. An epoch completes when all training pairs have been processed once. Early stopping terminates training if validation loss does not improve for five epochs, preventing overfitting.

### 3.6.4 Training Process

The training process iteratively adjusts network weights to minimize the loss function. Each iteration computes predictions for a batch of pairs, calculates the loss, and updates weights using backpropagation and gradient descent.

At the start of training, weights are initialized with random values. The first few predictions are essentially random, resulting in high loss. As training progresses, the network learns to produce better predictions, and loss decreases.

The training loop processes batches in shuffled order to ensure the network sees examples in varied sequence. For each batch, the algorithm computes predictions, calculates loss, and computes gradients. The optimizer updates weights in the direction that reduces loss.

Validation is performed after each epoch to monitor generalization. The validation set is never seen during training, so validation loss indicates how well the network applies to new images. If validation loss increases while training loss decreases, the network is overfitting.

The trained model with the lowest validation loss is saved as the best model. This ensures the deployed model has the best generalization rather than the model from the final epoch, which may have overfit.

### 3.6.5 Training Results

The training process achieved strong results on the face verification task. The training accuracy reached approximately ninety-nine point seven seven percent, indicating the network learned the training pairs extremely well.

The validation accuracy reached one hundred percent, suggesting the network generalized perfectly to the held-out validation set. This exceptional performance should be interpreted cautiously given the limited validation set size of three thousand pairs.

The training loss decreased from approximately point seven nine at the start to approximately point zero zero eight nine at the end. The validation loss similarly decreased to approximately point zero zero zero zero four. The small gap between training and validation losses indicates no significant overfitting.

These results suggest the network has learned effective representations for face verification. The transfer learning from MobileNetV2 combined with the custom training layers has produced a network that can distinguish same-person from different-person images with high accuracy.

---

## 3.7 Verification Pipeline

### 3.7.1 Verification Overview

Face verification determines whether two images show the same person. Unlike identification, which matches against a database of known persons, verification directly compares two images and returns a similarity score.

The verification pipeline is the core functionality of the system. When a user submits an image for recognition, the system must compare it against enrolled persons to determine identity. This comparison is verification.

The pipeline accepts a query image and a collection of enrolled persons with stored embeddings. It returns identification of the best matching person and the similarity score. Optionally, it can return scores for all enrolled persons for detailed analysis.

### 3.7.2 Query Processing

Query processing begins when a new image is submitted for verification. The image may come from webcam capture, file upload, or API request. The processing ensures the image is valid and can be processed.

The first step is saving the received image to a temporary file. The FastAPI endpoint receives the uploaded file and writes it to a temporary location. This enables subsequent processing functions to work with file paths rather than binary data.

The second step is preprocessing the query image. The face detection identifies the face region, cropping extracts the face, resizing normalizes the dimensions, and normalization scales the pixel values. The result is a ninety-six by ninety-six by three float32 array ready for the network.

The third step is extracting the query embedding. The preprocessed image is passed through the embedding network. The network outputs a two hundred fifty-six-dimensional embedding vector representing the query face.

### 3.7.3 Embedding Comparison

After extracting the query embedding, the system compares it against stored embeddings for enrolled persons. Each enrolled person has one or more stored embeddings from their enrollment images.

For persons with multiple enrolled images, the system compares against the average embedding. The average embedding represents the centroid of all enrollment images for that person. This averaging provides robustness to variations among enrollment images.

The comparison uses cosine similarity between the query embedding and each stored embedding. Cosine similarity is computed as the dot product of unit-normalized embeddings. Higher similarity indicates more likely match.

The system identifies the enrolled person with the highest similarity score. This is the best match candidate. The similarity score indicates the confidence of the match.

### 3.7.4 Decision Making

The final step is deciding whether the best match represents a genuine match or should be rejected as unknown. The threshold comparison determines this decision.

If the best match similarity exceeds the configured threshold, the system returns the matched person's name. The similarity score is included in the response, indicating the confidence of the match.

If the best match similarity does not exceed the threshold, the system returns unknown. This indicates the query face does not match any enrolled person with sufficient confidence.

The threshold provides a tunable parameter for trade-offs between security and convenience. Higher thresholds are more secure (fewer false accepts) but less convenient (more false rejects). The default threshold of point five provides reasonable balance.

### 3.7.5 Response Format

The verification response includes the result of the verification decision along with supporting information. This enables clients to understand and act on the result appropriately.

The response indicates whether the face was recognized, meaning the similarity exceeded the threshold. If recognized, the response includes the person's name as stored in the enrollment data.

The response includes the raw similarity score for the best match. This enables clients to make their own decisions based on the score, such as requiring higher confidence for sensitive operations.

The response includes the configured threshold so clients know what decision criteria were applied. It also includes the processing time, enabling performance monitoring and optimization.

---

## 3.8 Chapter Summary

This chapter has provided a comprehensive explanation of the face recognition system methodology, covering every major component from image input through verification output.

The face detection section explained how Haar cascade classifiers locate faces in images and how the implementation handles various detection scenarios including failures and multiple faces.

The image preprocessing section detailed the pipeline that transforms raw images into network-ready inputs, including face extraction, resizing, color conversion, and pixel normalization.

The Siamese neural network section described the complete architecture including the MobileNetV2 embedding network, the custom comparison network, and the custom layer implementations.

The similarity metrics section explained L1 distance and cosine similarity, including how they are computed and why combining both provides better results than either alone.

The training methodology section covered the data preparation, augmentation, configuration, and results of the model training process.

The verification pipeline section tied everything together, explaining how query images are processed, compared to enrolled persons, and turned into verification decisions.

With this understanding of the methodology, the report proceeds to Chapter 4, which presents the system design including architecture diagrams, data flows, and interface specifications.

---

**End of Chapter 3**
