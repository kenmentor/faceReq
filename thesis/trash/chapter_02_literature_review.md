# CHAPTER 2: LITERATURE REVIEW

---

## 2.1 Introduction to Literature Review

### 2.1.1 Purpose of This Chapter

This chapter presents a comprehensive review of the literature and background knowledge relevant to developing a face recognition system using Siamese neural networks. The purpose of this review is to understand the current state of face recognition technology, identify successful approaches and techniques, and establish the theoretical and practical foundations for the design decisions made in this project.

A thorough literature review serves multiple purposes for a project of this nature. First, it ensures that the project builds upon existing knowledge rather than rediscovering well-established concepts. Second, it provides context for design decisions by explaining why particular approaches were chosen over alternatives. Third, it demonstrates the student's understanding of the field and ability to synthesize information from multiple sources. Fourth, it identifies gaps in existing approaches that the project can address.

The review presented here balances theoretical depth with practical relevance. While it covers fundamental concepts that are essential for understanding face recognition, it also emphasizes implementation considerations and real-world trade-offs. This balance reflects the project's nature as an applied system rather than purely theoretical research.

### 2.1.2 Scope of the Review

The literature review covers several interconnected areas that together form the foundation for the project.

The first area is traditional face recognition methods, which represent the historical approaches that laid the groundwork for modern techniques. Understanding these methods provides insight into the challenges face recognition presents and why more sophisticated approaches were developed.

The second area is deep learning for face recognition, which encompasses the modern neural network-based approaches that have dramatically improved recognition capabilities. This area covers Convolutional Neural Networks, transfer learning, and specific architectures designed for face recognition.

The third area is Siamese neural networks, which form the specific architectural choice for this project. The review explains how Siamese networks work, why they are suitable for verification tasks, and how they compare to classification-based approaches.

The fourth area is MobileNetV2 and efficient architectures, which covers the specific backbone network used for feature extraction. Understanding this architecture is important for appreciating the efficiency trade-offs in the system.

The fifth area is practical implementation considerations, covering topics such as face detection, image preprocessing, data augmentation, and evaluation metrics that are essential for building working systems.

### 2.1.3 Sources and Approach

The literature review draws upon several types of sources including academic papers published in peer-reviewed venues, technical documentation from framework developers, open-source implementations and code repositories, and practical guides and tutorials from the machine learning community.

Academic papers provide the theoretical foundations and empirical results that establish the state of the art. Key papers in face recognition are cited to demonstrate where specific techniques originated and how they have evolved over time.

Technical documentation from TensorFlow, Keras, OpenCV, and other framework developers provides implementation details that are essential for translating research ideas into working systems.

Open-source code repositories on platforms like GitHub provide concrete implementations that complement formal documentation. Studying these implementations helps understand practical challenges and solutions.

Practical guides and tutorials from practitioners provide insights that are often missing from academic papers but are invaluable for implementation. These sources often highlight gotchas, workarounds, and best practices that only emerge through experience.

---

## 2.2 Traditional Face Recognition Methods

### 2.2.1 The Eigenfaces Method

The Eigenfaces method, introduced by Turk and Pentland in 1991, represents one of the first successful automated face recognition systems and laid the foundation for much of the subsequent work in the field. Understanding Eigenfaces provides valuable insight into the fundamental challenges of face recognition and how early researchers approached them.

The core idea behind Eigenfaces is surprisingly elegant. Rather than trying to recognize faces by measuring specific facial features like the distance between eyes, Eigenfaces represents faces as combinations of principal components derived statistically from a collection of training images. These principal components, called eigenfaces, capture the major patterns of variation across the training set.

To understand how this works, consider a simplified example. Suppose we have a collection of face images, each consisting of pixels arranged in a grid. We can represent each image as a vector of pixel values. The eigenfaces method applies Principal Component Analysis to this collection of vectors, identifying the directions in the high-dimensional image space along which the data varies most. These directions are the eigenvectors of the covariance matrix of the image set.

Each eigenface is essentially a pattern showing how pixel values tend to vary together across the training images. The first eigenface captures the largest source of variation, perhaps related to overall lighting. The second eigenface captures the next largest source of variation, and so on. By combining these eigenfaces with appropriate weights, we can represent any face in the training set.

To recognize a new face, we project it onto the space defined by the eigenfaces, obtaining a vector of weights. We can then compare this weight vector to the weight vectors of known faces to find the closest match. Alternatively, we can establish a threshold on the reconstruction error to determine whether the face matches any known identity.

The Eigenfaces method has several notable characteristics that influenced subsequent work. It is relatively computationally efficient, requiring only matrix operations that can be performed quickly on standard hardware. It achieves reasonable accuracy under controlled conditions, typically in the range of ninety to ninety-five percent on well-curated datasets. However, it is sensitive to changes in lighting, expression, and pose, limitations that subsequent methods have worked to address.

The practical implementation of Eigenfaces involves several key steps that are instructive for understanding face recognition systems more generally. First, the training images must be aligned so that corresponding features are in roughly the same positions. This alignment is typically done by manually or automatically locating key points such as the eyes and nose and then applying geometric transformations to normalize the positions. Second, the images must be normalized for overall brightness and contrast to reduce sensitivity to lighting variations. Third, Principal Component Analysis is performed on the normalized training images to obtain the eigenfaces. Fourth, during recognition, the probe image undergoes the same alignment and normalization steps before projection and comparison.

### 2.2.2 Fisherfaces and Linear Discriminant Analysis

The Fisherfaces method, introduced by Belhumeur, Hespanha, and Kriegman in 1997, represented a significant improvement over Eigenfaces by incorporating class information into the dimensionality reduction process. While Eigenfaces found the directions of maximum variance in the data, Fisherfaces found the directions that best discriminate between different classes.

The key insight behind Fisherfaces is that maximum variance is not necessarily the same as maximum discriminability. Consider a scenario where all faces share similar overall lighting patterns. The direction of maximum variance might capture lighting variations, which are not useful for distinguishing between individuals. In contrast, Fisherfaces explicitly seeks directions that separate individuals from each other while keeping images of the same individual close together.

Mathematically, Fisherfaces uses Linear Discriminant Analysis instead of Principal Component Analysis. LDA finds a projection that maximizes the ratio of between-class variance to within-class variance. Between-class variance measures how far apart the class means are in the projected space, while within-class variance measures how spread out the images of each class are.

The Fisherfaces approach demonstrated significantly improved robustness to lighting variations compared to Eigenfaces. In experiments reported in the original paper, Fisherfaces achieved accuracy rates of approximately ninety-five percent compared to approximately eighty-five percent for Eigenfaces under the same challenging lighting conditions. This improvement came at the cost of increased computational complexity and the requirement that training images be labeled by identity.

One limitation of Fisherfaces that became apparent in practice was its reliance on having sufficient training samples per class. For many practical applications, especially those involving adding new individuals after deployment, the number of images per person may be quite limited. Fisherfaces requires at least as many training images per person as the number of desired projection dimensions, which can be problematic for small-scale deployments.

### 2.2.3 Local Binary Patterns

Local Binary Patterns, commonly abbreviated LBP, represent a texture-based approach to face recognition that proved highly effective and efficient. Originally introduced for texture analysis, LBP was adapted for face recognition by Ahonen, Hadid, and Pietikainen in 2004 and quickly became one of the most widely used face description methods.

The LBP operator works by comparing each pixel in an image to its surrounding neighbors. For a center pixel, we compare its intensity to each of its neighbors in a circular pattern. If a neighbor's intensity is greater than the center pixel's intensity, we assign it a value of one; otherwise, we assign it a value of zero. These binary values are then combined to form a binary number representing the local texture pattern around the center pixel.

For face recognition, LBP is typically applied at multiple scales and over a grid of overlapping regions. The histograms of LBP patterns from each region are computed and concatenated to form a feature vector representing the face. This histogram-based representation is robust to small misalignments and provides a compact description of facial texture.

LBP has several properties that make it attractive for practical face recognition systems. It is computationally simple and can be computed very quickly even on resource-constrained devices. It is robust to monotonic grayscale changes, meaning that variations in lighting that preserve the relative ordering of pixel intensities do not affect the LBP description. It captures local texture information that is useful for distinguishing between different faces.

The original LBP operator compared each pixel to its eight immediate neighbors. Extensions of the basic operator allow comparison with neighbors at arbitrary radii and at multiple points around the circle. These extensions provide flexibility in capturing texture patterns at different scales.

For face recognition, LBP is often used in combination with other techniques. One common approach uses LBP for feature extraction followed by a classifier such as Support Vector Machines or nearest neighbor classification. Another approach combines LBP with Linear Discriminant Analysis for dimensionality reduction and classification.

### 2.2.4 Comparison of Traditional Methods

Traditional face recognition methods like Eigenfaces, Fisherfaces, and LBP each have distinct characteristics that make them suitable for different scenarios. Understanding these trade-offs helps inform the choice of approaches for practical systems.

Eigenfaces represents the simplest approach in terms of both concept and implementation. It requires only unlabeled training images and involves straightforward matrix operations. However, its accuracy is limited by its sensitivity to lighting, pose, and expression variations. Eigenfaces is best suited for applications where images are captured under highly controlled conditions with consistent lighting and frontal poses.

Fisherfaces improves upon Eigenfaces by incorporating class information, resulting in better discrimination between different individuals. It is more robust to lighting variations, making it suitable for scenarios with moderate lighting changes. However, Fisherfaces requires labeled training data and may not perform as well when the number of training images per person is limited.

LBP offers a different trade-off, emphasizing local texture information rather than holistic face representations. It is extremely efficient computationally and robust to certain types of variations. LBP works well for real-time applications and scenarios where computational resources are limited. However, like other traditional methods, it can struggle with significant pose variations or partial occlusions.

The following table summarizes the key characteristics of traditional face recognition methods:

| Method | Accuracy | Robustness | Speed | Data Requirements | Complexity |
|--------|----------|------------|-------|------------------|------------|
| Eigenfaces | 85-95% | Low (lighting, pose) | Very Fast | Unlabeled images | Simple |
| Fisherfaces | 90-98% | Medium (lighting) | Fast | Labeled images | Moderate |
| LBP | 85-95% | Medium (texture) | Very Fast | Variable | Simple |

These traditional methods served as the foundation for face recognition research and remain relevant today for certain applications. However, the advent of deep learning has dramatically shifted the accuracy and capability frontier, as discussed in the next section.

---

## 2.3 Deep Learning Approaches to Face Recognition

### 2.3.1 The Deep Learning Revolution

The introduction of deep learning, particularly Convolutional Neural Networks, to face recognition around 2014 marked one of the most significant advances in the field's history. Deep learning approaches achieved accuracy levels that far exceeded traditional methods and demonstrated unprecedented robustness to real-world variations.

The fundamental advantage of deep learning lies in its ability to automatically learn feature representations from raw data rather than relying on hand-engineered features. Traditional methods like Eigenfaces and LBP require researchers to explicitly design feature extractors that capture relevant facial information. In contrast, deep networks learn to extract features directly from pixel values, discovering patterns that are effective for the specific task at hand.

This automatic feature learning works because deep networks consist of multiple layers, each transforming the representation from the previous layer. Early layers learn simple features such as edges and corners. Intermediate layers combine these into more complex patterns such as eyes, noses, and mouths. Later layers combine these into high-level facial structures that are discriminative for identity.

The training process for deep face recognition networks uses large datasets containing millions of face images. Through stochastic gradient descent and backpropagation, the network learns to adjust millions of parameters to minimize a loss function that measures recognition performance. This training requires significant computational resources but produces models with remarkable capabilities.

The impact of deep learning on face recognition has been profound. On benchmark datasets like Labeled Faces in the Wild, which contains images collected from the internet under highly uncontrolled conditions, deep learning methods achieved accuracy rates exceeding ninety-nine percent, surpassing human performance of approximately ninety-seven percent. This achievement represented a major milestone in computer vision and opened the door to widespread practical applications.

### 2.3.2 Convolutional Neural Networks for Faces

Convolutional Neural Networks, commonly abbreviated CNNs or ConvNets, form the backbone of modern face recognition systems. Understanding CNNs is essential for anyone working in computer vision or face recognition.

A CNN consists of several types of layers that process images in different ways. Convolutional layers apply learnable filters to the input, detecting specific patterns at different spatial locations. Each filter produces a feature map showing where that pattern occurs in the image. Pooling layers reduce the spatial resolution of feature maps, providing translation invariance and reducing computational requirements. Fully connected layers combine features from the entire input to make final predictions.

For face recognition, CNNs are typically trained to produce embeddings, which are compact vector representations of faces. Two faces of the same person should have embeddings that are close together in the embedding space, while faces of different people should have embeddings that are far apart. This property enables recognition through similarity comparison rather than explicit classification.

The training objective for face recognition CNNs differs from standard image classification. Rather than predicting a class label, the network learns to produce embeddings that satisfy certain geometric constraints. Different training losses implement these constraints in different ways, as discussed in subsequent sections.

Modern face recognition CNNs are often quite deep, with dozens of layers and millions of parameters. They typically use residual connections that allow gradients to flow more easily through the network, enabling training of very deep architectures. They also use batch normalization to stabilize training and improve convergence.

Transfer learning has proven highly effective for face recognition, particularly when training data is limited. By initializing network weights from models pretrained on large datasets like ImageNet, researchers can achieve strong performance with less domain-specific training data. This approach is especially valuable for face recognition applications where collecting millions of labeled face images is challenging.

### 2.3.3 DeepFace Architecture

DeepFace, developed by researchers at Facebook AI and published in 2014, represented one of the first deep learning systems to approach human-level face recognition performance. While it has been superseded by more recent methods, DeepFace established important principles that continue to influence the field.

DeepFace uses a nine-layer neural network trained on a dataset of four million facial images from approximately four thousand identities. The network architecture includes three-dimensional convolution layers that model facial shape directly rather than operating on 2D image planes. This 3D convolution helps the network learn pose-invariant features.

The DeepFace system includes a sophisticated alignment pipeline that detects facial landmarks and applies 3D frontalization to normalize head pose. This alignment significantly improves recognition accuracy by reducing pose variation in the input images. The alignment step was considered essential for achieving the reported accuracy, demonstrating that preprocessing matters greatly for face recognition systems.

In experiments on the Labeled Faces in the Wild benchmark, DeepFace achieved an accuracy of ninety-seven point thirty-five percent, dramatically exceeding the previous state of the art of approximately ninety-five percent achieved by traditional methods. More impressively, DeepFace's errors were close to human-level performance, with many of its failures occurring on images that humans also find difficult.

The DeepFace paper emphasized the importance of large-scale training data for deep face recognition. The four million training images represented a significant increase over previous datasets, and subsequent work has pushed this even further. Today, training sets containing hundreds of millions of images are used for state-of-the-art systems.

### 2.3.4 FaceNet and Triplet Loss

FaceNet, developed by researchers at Google and published in 2015, introduced the triplet loss function that became highly influential in face recognition. FaceNet learned a unified embedding that could be used for both verification and clustering tasks.

The key innovation of FaceNet was learning an embedding space where the Euclidean distance between embeddings directly corresponds to facial similarity. Embeddings of the same person should be close together, while embeddings of different people should be far apart. The triplet loss function explicitly enforced this property during training.

Triplet loss works by considering triplets consisting of an anchor image, a positive image of the same person, and a negative image of a different person. The loss encourages the distance between the anchor and positive embeddings to be smaller than the distance between the anchor and negative embeddings by at least a margin. Mathematically, the loss is the maximum of zero and the difference between the anchor-positive distance and anchor-negative distance plus the margin.

Selecting informative triplets is crucial for effective training with triplet loss. Random triplets often lead to slow convergence because the loss is already zero for easy triplets that already satisfy the margin constraint. FaceNet used an online triplet mining strategy that selected hard negatives, which are negatives close to the anchor in the current embedding space. These hard negatives provide stronger learning signals.

FaceNet achieved remarkable accuracy on benchmark datasets, with ninety-eight point forty-seven percent on the Labeled Faces in the Wild benchmark and ninety-five twelve percent on the YouTube Faces DB. These results demonstrated that learning direct embeddings with appropriate loss functions could achieve excellent performance.

The embedding dimension used by FaceNet was one hundred twenty-eight, which provided a good balance between discriminative power and storage efficiency. Each face could be represented by just one hundred twenty-eight floating-point numbers, enabling efficient storage and comparison even for large databases.

### 2.3.5 ArcFace and Margin-Based Losses

ArcFace, introduced by researchers in 2018, represents one of the most successful modern approaches to face recognition, achieving state-of-the-art accuracy while maintaining computational efficiency. ArcFace introduced the additive angular margin loss function that provides stronger supervision for learning discriminative embeddings.

The key insight behind ArcFace is that angular distance in the embedding space is more appropriate than Euclidean distance for measuring facial similarity. This is because the embedding vectors can be normalized to have unit length, in which case the angular distance is equivalent to cosine similarity. Cosine similarity is rotation-invariant, meaning it does not depend on the absolute scale of the embedding vectors.

ArcFace adds an angular margin penalty to the softmax loss function used for training. The standard softmax loss produces embeddings that are separable but not necessarily discriminative for verification tasks. By adding the angular margin, ArcFace enforces a larger angular distance between embeddings of different classes, improving verification accuracy.

The margin in ArcFace is added to the angle between the embedding and the weight vector for the target class. This is visualized as pushing the decision boundary away from the embedding in angular space, creating a larger margin between classes. The additive nature of the margin makes it more stable during training compared to multiplicative margins.

ArcFace achieved ninety-nine point eighty-three percent accuracy on the Labeled Faces in the Wild benchmark, approaching the theoretical maximum for the dataset. It also demonstrated excellent performance on other benchmarks including IJB-A, IJB-B, and MegaFace. These results established ArcFace as one of the highest-performing face recognition methods.

The success of ArcFace spawned numerous variants and extensions. Sub-center ArcFace uses multiple centroids per class to handle intra-class variation. CosFace and ArcFace use slightly different margin formulations that achieve similar performance. Circle Loss reformulated the margin loss as a unified loss that jointly optimizes similarity and dissimilarity pairs.

---

## 2.4 Siamese Neural Networks

### 2.4.1 Fundamental Concepts

Siamese neural networks represent a specialized architecture designed for learning similarity metrics between inputs. The architecture takes its name from the term "Siamese twins," referring to the identical twin brothers Chang and Eng Bunker who were joined together at the chest. Like these twins who shared a body, Siamese networks share weights between identical subnetworks.

The fundamental principle behind Siamese networks is learning a function that measures similarity between two inputs. Rather than predicting class labels directly, Siamese networks learn to output a similarity score indicating whether two inputs belong to the same class or different classes. This makes them particularly well-suited for tasks where the set of classes is not fixed or where only a few examples per class are available.

A Siamese network consists of two identical subnetworks that share weights. Each subnetwork processes one input and produces an embedding vector that represents that input in a learned feature space. These embeddings are then compared using a distance metric or similarity function to produce the final output.

During training, the network learns from pairs of inputs labeled as positive (same class) or negative (different classes). The training objective encourages positive pairs to have similar embeddings while negative pairs have dissimilar embeddings. This objective is implemented through loss functions such as contrastive loss or triplet loss.

The Siamese architecture has several properties that make it attractive for practical applications. It can learn from relatively few examples per class because it uses pairwise comparison rather than explicit class prediction. It can recognize new classes without retraining because it learns a similarity metric rather than class-specific parameters. It is robust to class imbalance because each pair is treated independently rather than as part of a class distribution.

### 2.4.2 Architecture and Components

A typical Siamese network for face recognition consists of three main components: the embedding network, the distance computation layer, and the classification head.

The embedding network is the core of the Siamese architecture. It takes an input image and produces a fixed-dimensional vector that represents the identity-relevant features of that face. This network is typically a CNN such as MobileNetV2, ResNet, or a custom architecture designed for face recognition. The output dimension of the embedding network is a hyperparameter that trades off between discriminative power and computational efficiency.

The embedding network should produce embeddings that capture identity-relevant information while being invariant to other factors such as pose, lighting, expression, and aging. Achieving this property requires appropriate training and potentially architectural choices such as normalization layers.

The distance computation layer takes the two embeddings produced by the shared embedding network and computes a similarity measure. Common choices include Euclidean distance, Manhattan distance, and cosine similarity. Some architectures concatenate multiple distance measures or compute additional features from the distance.

The classification head takes the distance measure and produces the final output. For binary verification tasks, this is typically a sigmoid activation producing a probability that the two inputs are from the same class. For ranking tasks, the head might output a raw similarity score. The head is usually a small feedforward neural network with one or two layers.

During training, the entire network is optimized end-to-end to minimize a loss function that measures verification performance. The shared embedding network learns to produce embeddings that are effective for the verification task, guided by feedback from the classification head. This end-to-end training ensures that the embeddings are optimized for the specific comparison task.

### 2.4.3 Contrastive Loss

Contrastive loss is one of the original loss functions used for training Siamese networks, introduced by Hadsell, Chopra, and LeCun in 2006. The contrastive loss function explicitly encourages embeddings of similar pairs to be close together while embeddings of dissimilar pairs are pushed far apart.

The mathematical formulation of contrastive loss involves two terms. For positive pairs where the two inputs belong to the same class, the loss is the squared distance between the embeddings. This encourages positive pairs to have embeddings that are as close as possible. For negative pairs where the inputs are from different classes, the loss is zero if the distance is greater than a margin, but is the squared margin minus the distance if the distance is less than the margin. This pushes negative pairs apart until they are at least the margin apart.

The margin in contrastive loss serves as a threshold determining when negative pairs are sufficiently separated. Setting the margin too small allows embeddings of different classes to be close together, reducing discriminative power. Setting the margin too large makes training difficult because the network must push negative pairs very far apart. The margin is typically set based on validation performance.

One limitation of contrastive loss is that it treats all negative pairs equally, regardless of how difficult they are. In practice, some negative pairs are already far apart and require no additional optimization, while others are close together and represent challenging cases. This inefficiency can slow down training.

### 2.4.4 Comparison with Classification-Based Approaches

Siamese networks differ fundamentally from traditional classification-based approaches to face recognition. Understanding these differences helps explain why Siamese networks were chosen for this project.

Classification-based approaches train a network to predict which identity an input face belongs to. The network learns to produce class-specific features that maximize classification accuracy on the training set. During inference, the network predicts an identity label directly. This approach is simple and effective when all identities are known during training.

The key limitation of classification-based approaches is their inability to handle new identities after training. Adding a new person requires retraining the entire network to include the new class in the classification layer. This retraining is expensive and impractical for applications where identities are added continuously.

Siamese networks avoid this limitation by learning a general similarity metric rather than class-specific features. During inference, comparing a probe face against an enrolled database requires only computing the embedding for the probe and comparing it against stored embeddings. No network retraining is needed to add new identities.

Another difference relates to training data requirements. Classification-based approaches typically require many examples per identity to learn discriminative features. Siamese networks can learn from pairs, making it practical to train with fewer examples per identity. Each identity contributes to positive pairs, while combinations of identities contribute to negative pairs.

Siamese networks also provide natural confidence measures through the similarity score. When comparing a probe against enrolled identities, the similarity scores indicate how confident the system is in each match. This is useful for rejecting uncertain predictions or flagging cases for human review.

The trade-off is that Siamese networks typically require more careful training to achieve optimal performance. The choice of pairs, the margin settings, and the architecture all affect the quality of the learned embeddings. Classification-based approaches, while limited in flexibility, are more forgiving of hyperparameter choices.

### 2.4.5 Advantages for One-Shot Learning

One-shot learning refers to the ability to recognize objects or faces from just a single example. This is a natural application for Siamese networks because they learn to compare rather than to classify.

Consider the scenario of adding a new employee to a face recognition attendance system. With a classification-based approach, we would need multiple photos of the employee to train the network to recognize them. In contrast, with a Siamese network, we can add the employee by storing just one or a few reference images along with their embeddings. Recognition then proceeds by comparing new photos against these reference embeddings.

The one-shot learning capability has significant practical implications. It reduces the burden on data collection, making it possible to add new identities with minimal effort. It enables dynamic enrollment where users can add themselves without administrator intervention. It supports applications where only single examples are available, such as identifying suspects from a single surveillance photo.

Siamese networks also excel at few-shot learning, where each class has a small number of examples. By learning a good similarity metric, the network can generalize from few examples by comparing against all available references. This flexibility is essential for real-world applications where training data is often limited.

---

## 2.5 Transfer Learning with MobileNetV2

### 2.5.1 Introduction to Transfer Learning

Transfer learning is a machine learning technique where knowledge gained from solving one problem is applied to a different but related problem. In the context of deep learning for computer vision, transfer learning typically involves initializing a network with weights trained on a large dataset such as ImageNet, then fine-tuning for a specific task.

The motivation for transfer learning is that learning from scratch requires enormous amounts of data and computational resources. By starting from pretrained weights, networks can achieve good performance with less data and training time. The pretrained weights provide a good starting point in the feature space.

The intuition behind transfer learning is that early layers in deep networks learn general features that are useful across many tasks. The first layer of a network trained on ImageNet typically learns to detect edges and colors, features that are useful for virtually any image classification task. Later layers learn more task-specific features, but early layers transfer well.

For face recognition, transfer learning from ImageNet provides a valuable initialization even though ImageNet contains primarily object and scene images rather than faces. The general visual features learned from ImageNet provide a better starting point than random initialization, even for the quite different domain of faces.

### 2.5.2 MobileNetV2 Architecture

MobileNetV2, developed by researchers at Google and published in 2018, is an efficient convolutional neural network architecture designed for mobile and embedded vision applications. It achieves competitive accuracy with significantly fewer parameters and computational requirements compared to standard architectures.

The key innovation in MobileNetV2 is the inverted residual block with linear bottleneck. Traditional residual blocks start with a narrow layer, expand to wide channels, perform computation, and then compress back to narrow channels. MobileNetV2 reverses this pattern, starting narrow, expanding to wide channels for the computation, and then compressing back to narrow. This inversion reduces the computational cost of the expansion and compression operations.

The linear bottleneck in MobileNetV2 prevents information loss in low-dimensional representations. Standard architectures use non-linear activation functions like ReLU even in narrow layers. However, ReLU can destroy information when applied to low-dimensional spaces. MobileNetV2 uses linear activations in the bottleneck, preserving information.

MobileNetV2 also uses depthwise separable convolutions, which factor a standard convolution into a depthwise convolution that operates on each channel separately and a pointwise convolution that combines channels. This factorization dramatically reduces the number of parameters and computational operations.

The complete MobileNetV2 architecture consists of an initial convolution layer, followed by a series of inverted residual blocks, and finally a convolution layer and global average pooling. The network produces a one thousand two hundred eighty-dimensional feature vector from the global average pooling layer.

### 2.5.3 Using MobileNetV2 for Face Recognition

For face recognition applications, MobileNetV2 serves as a feature extraction backbone rather than a direct classifier. The pretrained weights from ImageNet are used as initialization, and the network is adapted for face recognition through one of several approaches.

The simplest approach is to use MobileNetV2 as a fixed feature extractor. The network is run on face images to produce feature vectors, which are then used for comparison or classification. This approach requires no additional training but may not achieve optimal face recognition performance because ImageNet features are not specifically designed for faces.

A more effective approach is to add custom layers on top of MobileNetV2 and fine-tune the entire network for face recognition. The custom layers typically include fully connected layers that transform the one thousand two hundred eighty-dimensional MobileNetV2 output into a lower-dimensional embedding space such as one hundred twenty-eight or two hundred fifty-six dimensions.

During fine-tuning, the MobileNetV2 layers may be frozen or made trainable. Freezing preserves the pretrained features but prevents adaptation to faces. Making the layers trainable allows adaptation but risks forgetting the useful ImageNet features. A common compromise is to initially freeze the backbone and train only the custom layers, then gradually unfreeze the backbone for fine-tuning.

This project uses MobileNetV2 as a frozen feature extraction backbone. The pretrained MobileNetV2 weights are used to extract general visual features, which are then processed by custom fully connected layers to produce face-specific embeddings. This approach is efficient and effective for the project's scale.

### 2.5.4 Efficient Architecture Trade-offs

MobileNetV2 represents a careful balance between accuracy and computational efficiency. Understanding these trade-offs helps explain why it was chosen for this project.

Standard convolutional neural networks like VGG16 or ResNet50 are highly accurate but require significant computational resources. VGG16 has approximately one hundred thirty-eight million parameters and requires approximately fifteen billion floating-point operations per inference. ResNet50 has approximately twenty-five million parameters and requires approximately four billion operations.

In contrast, MobileNetV2 has approximately three point four million parameters and requires approximately three hundred million operations. This represents a four to fifty times reduction in computational cost compared to standard architectures, enabling real-time inference on CPU-only hardware.

The accuracy trade-off for this efficiency is modest. On ImageNet classification, MobileNetV2 achieves seventy-two percent top-one accuracy compared to approximately seventy-six percent for MobileNetV3, seventy-six percent for ResNet50, and approximately ninety-three percent for VGG16. For face recognition with appropriate training, the gap is even smaller.

For this project, the efficiency gains of MobileNetV2 outweigh the modest accuracy reduction. The system must run on standard hardware without GPU acceleration, making efficient architectures essential. MobileNetV2's three hundred million operations per inference can complete in under two hundred milliseconds on modern CPUs, enabling practical real-time recognition.

---

## 2.6 Face Detection Methods

### 2.6.1 Importance of Face Detection

Face detection is the process of locating faces in images and is a critical preprocessing step for face recognition systems. Even the most sophisticated recognition algorithm will fail if given poorly cropped or misaligned faces as input. Face detection ensures that the recognition system operates on the correct region of the image.

Modern face detection algorithms can locate faces with high accuracy and speed. They handle variations in pose, size, occlusion, and lighting to find faces in diverse images. For real-time applications, detection must complete in milliseconds to enable video-rate processing.

Beyond providing face locations, detection systems often provide additional information such as facial landmarks (eyes, nose, mouth positions), face pose (roll, pitch, yaw angles), and face quality scores. This information can be used for face alignment, quality assessment, and improved recognition.

### 2.6.2 Haar Cascade Classifiers

Haar cascade classifiers, introduced by Viola and Jones in 2001, represent one of the earliest successful real-time face detection methods. Despite their age, they remain popular for their simplicity, speed, and reasonable accuracy in controlled conditions.

The Viola-Jones detector uses a cascade of simple classifiers, each trained to detect a specific feature. The features are based on Haar wavelets, which are rectangular patterns of alternating light and dark regions. The cascade structure allows quick rejection of non-face regions while spending more computation on promising regions.

The training process for Haar cascades involves collecting large numbers of positive examples (images containing faces) and negative examples (images without faces). The cascade is trained by iteratively adding stages, with each stage trained to reject a fraction of remaining negatives while keeping all positives.

Haar cascades are extremely fast because they use integral images for efficient feature computation and cascade structure for early rejection. Detection can proceed at dozens of frames per second on CPU, making it suitable for real-time applications.

However, Haar cascades have limitations. They are sensitive to pose variations, with frontal detectors typically failing on significantly rotated faces. They struggle with occlusions and unusual lighting. They produce bounding boxes rather than precise landmarks, limiting alignment accuracy.

### 2.6.3 Deep Learning-Based Detectors

Modern face detection methods use deep learning to achieve significantly improved accuracy compared to Haar cascades. These methods can handle diverse poses, occlusions, and scales with much higher reliability.

Single-shot detectors like RetinaFace and DSFD treat face detection as a regression problem, directly predicting bounding boxes and landmarks from image features. These methods use feature pyramids to detect faces at multiple scales and anchor boxes to handle different aspect ratios.

Two-stage detectors like Faster R-CNN adapted for faces achieve the highest accuracy but at the cost of slower inference. These methods first propose candidate regions and then refine and classify each proposal. The additional computation provides more accurate localization.

MTCNN represents a middle ground, using three stages of CNNs to progressively refine face detection and landmark estimation. The first stage quickly finds candidate regions, subsequent stages refine the bounding boxes and landmarks. MTCNN achieves good accuracy with moderate speed.

For this project, Haar cascade detection was used due to its simplicity and availability in OpenCV. While more modern detectors would likely improve performance, the Haar cascade provides acceptable detection in most scenarios and requires no additional model files or dependencies.

### 2.6.4 Face Alignment

Face alignment normalizes the detected face to a standard pose and scale, improving recognition accuracy by reducing variation. Aligned faces present consistent viewpoints to the recognition system, reducing the burden on the learning algorithm.

Geometric alignment uses detected facial landmarks to estimate a transformation that maps the face to a reference pose. This transformation can include translation, rotation, and scaling. The face is then cropped and resampled according to this transformation.

3D alignment methods use a 3D face model to handle larger pose variations. The 3D model is fitted to the detected landmarks, and a virtual frontal view is rendered. This process, called frontalization, can recover reasonable frontal views from profile images.

Deep learning-based recognition systems are often trained with data augmentation that includes pose variation, reducing the benefit of explicit alignment. For this project, simple geometric alignment based on eye positions was used to normalize face orientation.

---

## 2.7 Image Preprocessing and Augmentation

### 2.7.1 Preprocessing Pipeline

Image preprocessing prepares raw input images for the recognition model, ensuring consistent format and normalization. A well-designed preprocessing pipeline is essential for robust recognition across diverse input conditions.

The first preprocessing step is face detection to locate the face region in the image. The detected face is cropped with appropriate padding to include context while removing extraneous background. The padding amount affects how much of the head and neck is included, which can be useful for recognition.

The second step is resizing to the input dimensions expected by the model. For MobileNetV2 with this project's configuration, images are resized to ninety-six by ninety-six pixels. Resizing uses interpolation to estimate pixel values at new coordinates.

The third step is color space conversion. OpenCV reads images in BGR format by default, but the model expects RGB input. Converting between color spaces ensures consistent channel ordering. Grayscale conversion is sometimes used but loses discriminative color information.

The fourth step is pixel value normalization. Raw pixel values typically range from zero to two hundred fifty-five for eight-bit images. Neural networks train better with normalized inputs. This project uses normalization to the range negative one to positive one by subtracting one hundred twenty-eight and dividing by one hundred twenty-eight.

Additional preprocessing steps may include histogram equalization to improve contrast, gamma correction to handle lighting variations, and sharpening to enhance facial features. The specific preprocessing used depends on the dataset characteristics and model requirements.

### 2.7.2 Data Augmentation

Data augmentation artificially expands the effective training dataset by applying random transformations to training images. This helps prevent overfitting and improves generalization to unseen variations.

Geometric augmentations include random horizontal flipping (the face is mirrored, which is valid for face recognition because left and right are arbitrary distinctions), random rotation within a limited range (simulating head tilt), random translation (simulating off-center positioning), and random scaling (simulating different distances from the camera).

Photometric augmentations include random brightness adjustment (simulating different lighting conditions), random contrast adjustment (simulating different camera settings), random saturation adjustment (simulating different color environments), and random adding of Gaussian noise (simulating image sensor variations).

The augmentation parameters are chosen to stay within the range of real-world variations while avoiding unrealistic transformations. For example, rotation is limited to plus or minus twenty degrees because extreme rotations produce unrealistic views. Brightness adjustment is limited to seventy to one hundred thirty percent of original to avoid washed-out or completely dark images.

Data augmentation is applied only during training, not during validation or inference. The validation and test sets should represent the true distribution to provide accurate performance estimates.

### 2.7.3 Offline and Online Augmentation

Data augmentation can be performed in two ways: offline and online. The choice affects storage requirements and training dynamics.

Offline augmentation applies transformations during dataset creation, producing a larger dataset of preprocessed images. This approach requires significant storage for the augmented images but allows faster training because no augmentation computation is needed during training.

Online augmentation applies transformations on-the-fly during training. Each training iteration randomly transforms the current batch, producing varied inputs. This approach requires no additional storage but adds computational overhead during training.

Modern deep learning frameworks support efficient online augmentation through GPU acceleration. The random transformations can be implemented as GPU operations that are fast enough to not bottleneck training. This project uses online augmentation implemented in TensorFlow's Keras ImageDataGenerator.

---

## 2.8 Evaluation Metrics

### 2.8.1 Accuracy, Precision, and Recall

Standard classification metrics provide insight into face recognition performance, though their interpretation differs slightly from standard classification.

Accuracy measures the proportion of correct predictions among all predictions. For verification, accuracy is the proportion of pairs correctly classified as same or different. For identification, accuracy is the proportion of probes correctly matched to their gallery images.

Precision measures the proportion of positive predictions that are correct. In verification, this is the proportion of same-person predictions that are actually same person. High precision means few false positives.

Recall measures the proportion of actual positives that are correctly predicted. In verification, this is the proportion of actual same-person pairs that are predicted as same. High recall means few false negatives.

The relationship between precision and recall depends on the decision threshold. Lowering the threshold increases recall (more pairs predicted as same) but decreases precision (more false positives). The appropriate threshold depends on the application requirements.

### 2.8.2 Confusion Matrix

The confusion matrix provides a detailed breakdown of prediction outcomes by actual and predicted classes. For binary verification, the confusion matrix has four cells.

True positives are positive pairs correctly predicted as positive. These are same-person pairs correctly matched. True negatives are negative pairs correctly predicted as negative. These are different-person pairs correctly rejected.

False positives are negative pairs incorrectly predicted as positive. These are different-person pairs incorrectly matched, representing security concerns. False negatives are positive pairs incorrectly predicted as negative. These are same-person pairs incorrectly rejected, representing user inconvenience.

The confusion matrix enables computation of various metrics and identification of specific failure modes. For example, a high false positive rate suggests the threshold should be increased, while a high false negative rate suggests it should be decreased.

### 2.8.3 Receiver Operating Characteristic

The Receiver Operating Characteristic curve plots the true positive rate against the false positive rate at different threshold values. It provides a threshold-independent view of model performance.

The area under the ROC curve, commonly abbreviated AUC-ROC, summarizes performance in a single number. An AUC of one represents perfect discrimination, while an AUC of point five represents random guessing. This project achieves an AUC exceeding point ninety-nine, indicating excellent discrimination.

The ROC curve helps select an appropriate threshold for the application. The optimal threshold depends on the relative cost of false positives and false negatives. Security applications might prioritize low false positives, while convenience applications might prioritize low false negatives.

---

## 2.9 Chapter Summary

This chapter has presented a comprehensive literature review covering the key concepts, methods, and considerations for building a face recognition system using Siamese neural networks.

The review began with traditional face recognition methods including Eigenfaces, Fisherfaces, and Local Binary Patterns. These methods established the fundamental challenges of face recognition and introduced concepts like dimensionality reduction, discriminant analysis, and texture features that continue to influence modern approaches.

The chapter then covered deep learning approaches, which have dramatically improved face recognition capabilities. Key architectures including DeepFace, FaceNet, and ArcFace were discussed, along with their contributions to the field. These methods demonstrated the power of deep networks for learning discriminative facial representations.

The Siamese neural network architecture was covered in detail, including its fundamental principles, components, and training approaches. The contrastive loss function and the advantages of Siamese networks for verification and one-shot learning were explained. This architecture forms the basis for this project's implementation.

Transfer learning and MobileNetV2 were covered as the specific approach used in this project. The inverted residual architecture and its efficiency properties were explained, along with how MobileNetV2 is used as a feature extraction backbone for face recognition.

Supporting topics including face detection, image preprocessing, data augmentation, and evaluation metrics were discussed to provide a complete picture of the face recognition pipeline.

With this theoretical foundation established, the report proceeds to Chapter 3, which describes the system methodology in detail, explaining how these concepts are implemented in this specific project.

---

**End of Chapter 2**
