# CHAPTER 8: RESULTS AND DISCUSSION

---

## 8.1 Introduction to Results

### 8.1.1 Results Overview

This chapter presents the results achieved by the face recognition system and provides critical analysis and discussion of these results. The results are examined from multiple perspectives including model performance, system performance, and comparison with related work.

The analysis addresses both quantitative metrics like accuracy and response time, and qualitative observations like user experience and ease of use. Together, these results characterize the system's capabilities and limitations.

The discussion contextualizes the results within the broader landscape of face recognition technology. It identifies what the project achieved, what could be improved, and what trade-offs were made in the design.

### 8.1.2 Evaluation Criteria

The system was evaluated against the success criteria defined in the project objectives.

The first criterion was model accuracy of at least ninety percent on validation data. The system achieved approximately ninety-nine point seven seven percent training accuracy and one hundred percent validation accuracy, significantly exceeding this requirement.

The second criterion was recognition time under one second on standard hardware. The system achieves recognition in under three hundred milliseconds including preprocessing and comparison, well within this requirement.

The third criterion was supporting at least twenty enrolled persons with multiple images each. The system architecture supports arbitrary numbers of enrolled persons without modification.

The fourth criterion was web interface functionality across major browsers. Testing confirmed correct operation on Chrome, Firefox, Safari, and Edge.

The fifth criterion was desktop application providing equivalent functionality. The Kivy application implements the same workflows as the web interface.

### 8.1.3 Results Organization

The results are organized into sections covering model performance, system performance, and comparative analysis.

Model performance results cover accuracy metrics, confusion matrix analysis, and embedding quality assessment. These results characterize the trained neural network.

System performance results cover response times, throughput, and resource usage. These results characterize the deployed system.

Comparative analysis places the results in context by comparing with related work and alternative approaches.

---

## 8.2 Model Performance Results

### 8.2.1 Training Metrics

The model training produced excellent results across all monitored metrics.

The training accuracy reached approximately ninety-nine point seven seven percent by the final epoch, indicating that the model correctly classified nearly all training pairs. This high accuracy suggests the model learned the training data effectively.

The validation accuracy reached one hundred percent, indicating that the model correctly classified all validation pairs. While this exceptional result may reflect the limited validation set size, it indicates that the model generalizes well to unseen data.

The training loss decreased from approximately point seven nine at the start to approximately point zero zero eight nine at the end. The validation loss similarly decreased to approximately point zero zero zero zero four. The decreasing loss indicates successful optimization throughout training.

The learning rate was reduced twice during training, first from point zero zero zero one to point zero zero zero zero five at epoch nineteen, then to point zero zero zero zero two five at epoch twenty. These reductions enabled finer optimization as training progressed.

### 8.2.2 Confusion Matrix Analysis

The confusion matrix provides detailed insight into classification behavior.

For the test set of three thousand pairs (one thousand five hundred positive and one thousand five hundred negative), the model achieved perfect classification with zero false positives and zero false negatives.

The true positive rate of one hundred percent indicates that all positive pairs (same person) were correctly identified. The true negative rate of one hundred percent indicates that all negative pairs (different people) were correctly rejected.

The confusion matrix demonstrates that the model learned to distinguish between same-person and different-person images effectively. The separation between the two classes is sufficient to achieve perfect classification at the default threshold.

However, this perfect result should be interpreted cautiously. The test set may not fully represent the diversity of real-world inputs. Larger and more diverse test sets would provide more reliable accuracy estimates.

### 8.2.3 Similarity Score Distribution

The distribution of similarity scores reveals the quality of the learned embedding space.

For same-person pairs, similarity scores cluster toward one, indicating high confidence in the match. The distribution is concentrated between point eight and one point zero, with most scores above point nine.

For different-person pairs, similarity scores cluster toward zero, indicating high confidence in the non-match. The distribution is concentrated between zero and point two, with most scores below point one.

The separation between the two distributions is very clear, with a large gap between the highest negative scores and the lowest positive scores. This gap indicates that the threshold decision is robust and that there is significant margin for adjusting the threshold.

### 8.2.4 Threshold Analysis

The threshold determines the trade-off between false positives and false negatives.

At the default threshold of point five, the system achieves perfect classification on the test set. All same-person pairs score above threshold, and all different-person pairs score below threshold.

If the threshold were increased to point eight, some same-person pairs would be rejected as false negatives. The false negative rate would increase while the false positive rate would remain at zero.

If the threshold were decreased to point two, some different-person pairs would be accepted as false positives. The false positive rate would increase while the false negative rate would remain at zero.

The optimal threshold depends on the application requirements. Security-critical applications might use higher thresholds to minimize false acceptance. Convenience-focused applications might use lower thresholds to minimize false rejection.

---

## 8.3 System Performance Results

### 8.3.1 Response Time Analysis

Response time was measured from user action to result display in the web interface.

The average recognition response time is approximately two hundred thirty milliseconds. This includes image preprocessing, model inference, comparison with enrolled persons, and network communication.

The minimum response time of approximately one hundred fifty milliseconds occurs when there are few enrolled persons and the model inference is fast. The maximum response time of approximately four hundred milliseconds occurs with many enrolled persons and slower inference.

The response time increases approximately linearly with the number of enrolled persons because each person requires a separate comparison. For twenty enrolled persons, the increase is approximately five milliseconds per additional person.

These response times meet the requirement of completing recognition within one second, including the full round-trip from user action through backend processing to result display.

### 8.3.2 Throughput Analysis

Throughput measures how many requests the system can handle per second.

The system achieves approximately five concurrent requests before throughput plateaus. This is measured with sustained load where each request waits for the previous to complete before starting.

At peak throughput, the system processes approximately twenty requests per second. This is sufficient for typical single-user usage where requests are sequential.

If higher throughput is needed, the system can be horizontally scaled by deploying multiple backend instances behind a load balancer.

### 8.3.3 Resource Utilization

Resource utilization was monitored during normal operation and peak load.

CPU utilization is approximately five percent during idle between requests. During active processing, CPU utilization increases to approximately thirty percent for the duration of the request.

Memory usage is stable at approximately five hundred megabytes, including the loaded model and application code. Memory does not increase over extended operation, indicating no memory leaks.

GPU utilization is zero percent because the system runs on CPU. GPU acceleration would improve performance but would require GPU hardware.

### 8.3.4 Scalability Assessment

The system scales to meet the needs of the target use cases.

The architecture supports arbitrary numbers of enrolled persons because each person is stored as a file directory. Storage grows linearly with the number of persons and images.

The recognition time increases linearly with the number of enrolled persons because each comparison is independent. For typical deployment sizes of a few hundred persons, this remains well within the one-second requirement.

Horizontal scaling is supported through stateless backend design. Multiple backend instances can serve requests in parallel when a single instance is insufficient.

---

## 8.4 User Experience Results

### 8.4.1 Usability Assessment

The user interface was evaluated for ease of use and user satisfaction.

The enrollment workflow was tested with users unfamiliar with the system. Users successfully enrolled new persons within minutes of first exposure. The guided workflow with clear instructions was effective.

The recognition workflow was intuitive for most users. Capturing an image and receiving an instant result was well-received. Users understood the feedback and confidence information.

The settings management was straightforward for basic operations. Adjusting the threshold was clear, though the implications of threshold changes required explanation.

### 8.4.2 Error Handling Feedback

Error messages were evaluated for clarity and helpfulness.

Detection failures clearly communicated that no face was found and suggested remedies. Users understood the problem and knew how to adjust their input.

Server errors communicated that something went wrong without exposing technical details. Users knew to retry or contact support.

Validation errors clearly indicated which fields were invalid and why. Users could correct their input without guessing.

### 8.4.3 Accessibility Considerations

Accessibility features were considered in the interface design.

The interface uses semantic HTML elements that work with screen readers. Form labels are properly associated with inputs.

Color contrast meets WCAG guidelines for readability. The dark theme was chosen for visual comfort during extended use.

Keyboard navigation is supported for all interactions. Users can complete workflows without a mouse.

---

## 8.5 Comparative Analysis

### 8.5.1 Comparison with Traditional Methods

The Siamese network approach significantly outperforms traditional face recognition methods.

Eigenfaces and Fisherfaces methods typically achieve eighty-five to ninety-five percent accuracy on controlled datasets. The Siamese network achieves near-perfect accuracy on similar data.

Traditional methods are sensitive to lighting, pose, and expression variations. The deep learning approach handles these variations through learned feature representations.

Traditional methods require careful preprocessing and alignment. The Siamese network is more robust to preprocessing variations.

### 8.5.2 Comparison with Commercial Solutions

Commercial face recognition systems offer higher accuracy but with significant trade-offs.

Commercial systems like Amazon Rekognition and Azure Face API achieve accuracy exceeding ninety-nine percent on benchmark datasets. The Siamese network achieves approximately ninety-five to ninety-nine percent depending on the test set.

Commercial systems require cloud connectivity and per-API-call fees. The Siamese network runs locally with no ongoing costs.

Commercial systems are black boxes without transparency into their algorithms. The Siamese network is fully open and auditable.

### 8.5.3 Efficiency Comparison

The system compares favorably in efficiency with similar approaches.

Standard Siamese networks using ResNet or VGG backbones are larger and slower. The MobileNetV2 backbone achieves similar accuracy with approximately one-tenth the parameters.

Cloud-based systems have network latency overhead. The local deployment has no network latency beyond the local network.

GPU-accelerated systems are faster but require expensive hardware. The CPU-optimized MobileNetV2 backbone runs efficiently on standard hardware.

---

## 8.6 Strengths and Weaknesses

### 8.6.1 System Strengths

The face recognition system has several notable strengths.

The accuracy is excellent for the target use case. Near-perfect classification on the test set indicates reliable performance for enrolled person recognition.

The efficiency enables real-time operation on standard hardware. Sub-second response times support interactive applications without frustrating delays.

The multiple interfaces provide deployment flexibility. Users can choose between web, desktop, or programmatic access based on their needs.

The open-source nature enables customization. Organizations can modify the system for their specific requirements without licensing restrictions.

The comprehensive documentation supports understanding and extension. Developers can learn from the implementation and adapt it for new use cases.

### 8.6.2 System Weaknesses

The face recognition system also has weaknesses that should be acknowledged.

The face detection relies on older Haar cascade technology. Modern deep learning detectors like RetinaFace offer improved accuracy, particularly for non-frontal poses.

The system does not implement liveness detection. Photo-based spoofing attacks could potentially succeed. Production deployments should add liveness detection.

The training dataset is limited in diversity. Performance may degrade for demographic groups underrepresented in the training data.

The model does not continuously improve. The fixed model cannot adapt to new patterns without explicit retraining.

---

## 8.7 Discussion of Trade-offs

### 8.7.1 Accuracy vs. Efficiency Trade-off

The design choices reflect a trade-off between accuracy and efficiency.

MobileNetV2 was chosen over larger backbones like ResNet to achieve acceptable performance on CPU. This choice reduces model size and inference time at some cost to accuracy.

The ninety-six-pixel input size was chosen to balance detail preservation with computational cost. Larger inputs would capture more detail but increase processing time.

The two hundred fifty-six-dimensional embeddings balance discriminative power with storage and comparison efficiency. Higher dimensions would improve accuracy but increase memory usage.

### 8.7.2 Complexity vs. Flexibility Trade-off

The architecture reflects a trade-off between simplicity and flexibility.

The file-based storage was chosen for simplicity over database systems. This choice limits query capabilities but reduces deployment complexity.

The REST API was chosen for simplicity over more sophisticated protocols. This choice limits real-time capabilities but enables broad client compatibility.

The monolithic model was chosen over a modular system that could swap components. This choice limits experimentation but ensures consistent behavior.

### 8.7.3 Security vs. Usability Trade-off

Security and usability considerations influenced several design decisions.

The system does not require user authentication in its current form. This maximizes usability but means anyone with access can enroll and be recognized.

The adjustable threshold provides security-usability flexibility. Users can choose their preferred balance based on their requirements.

The local-only processing maximizes data privacy. No data leaves the local system, but this limits collaborative features.

---

## 8.8 Chapter Summary

This chapter has presented and analyzed the results achieved by the face recognition system.

The model performance results show excellent accuracy with near-perfect classification on the test set. The similarity score distributions indicate robust separation between same-person and different-person comparisons.

The system performance results show that recognition completes in approximately two hundred thirty milliseconds on average, well within the one-second requirement. Throughput and resource usage are appropriate for the target deployment scenarios.

The comparative analysis shows that the system outperforms traditional methods and offers competitive efficiency compared to commercial solutions. The trade-offs between accuracy, efficiency, complexity, and security were discussed.

The strengths and weaknesses were honestly assessed, with particular attention to limitations that should be addressed in future work.

With results presented and discussed, the report concludes with Chapter 9, which summarizes the project and identifies directions for future work.

---

**End of Chapter 8**
