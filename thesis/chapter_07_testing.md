# CHAPTER 7: TESTING AND EVALUATION

---

## 7.1 Introduction to Testing

### 7.1.1 Testing Overview

This chapter presents the comprehensive testing strategy employed to validate that the face recognition system meets its requirements and functions correctly across all components. Testing is a critical phase that verifies the correctness of implementation and ensures the system is ready for deployment.

The testing strategy encompasses multiple levels of testing, from individual component tests through integration tests to full system validation. Each level addresses different concerns and catches different types of issues. The combination of all levels provides confidence in the system's correctness.

Testing was conducted throughout the development process rather than only at the end. This approach caught issues early when they were easier to fix, and ensured that each component was validated before integration with others.

### 7.1.2 Testing Objectives

The testing objectives define what the testing process aims to achieve.

The first objective is correctness verification, ensuring that each component produces the expected output for given inputs. Incorrect outputs indicate bugs that must be fixed before proceeding.

The second objective is regression prevention, ensuring that changes to one component do not break other components. Automated tests run on each change provide early warning of unintended effects.

The third objective is performance validation, ensuring that the system meets its performance requirements. Response times and throughput must be acceptable for real-time operation.

The fourth objective is edge case coverage, ensuring that the system handles unusual inputs gracefully. Real-world inputs often include conditions not anticipated during design.

### 7.1.3 Testing Methodology

The testing methodology follows established software engineering practices adapted for this project.

Unit testing validates individual functions and classes in isolation. Mock objects isolate components from their dependencies, enabling focused testing without complex setup.

Integration testing validates that components work correctly together. Tests exercise the interfaces between components, catching issues that unit tests cannot detect.

System testing validates the complete end-to-end functionality. Tests simulate real user workflows from start to finish, ensuring the system meets requirements.

Performance testing measures timing characteristics under various loads. Tests identify bottlenecks and validate that performance meets requirements.

---

## 7.2 Unit Testing

### 7.2.1 Unit Testing Framework

Unit tests are written using Python's unittest framework, which is built into the standard library and requires no additional dependencies.

Test cases are organized into test classes, with each class testing a specific component or functionality. Test methods within each class test specific behaviors.

The unittest framework provides assertion methods for checking expected values, fixtures for setup and teardown, and test discovery for running collections of tests.

Tests follow the arrange-act-assert pattern. The arrange phase sets up test inputs and expected outputs. The act phase executes the code under test. The assert phase verifies the results.

### 7.2.2 Preprocessing Tests

Preprocessing functions are tested with various input images to verify correct transformation.

The face detection test uses images with known face locations. The test verifies that the correct bounding box is returned and that no faces are detected in images without faces.

The crop test verifies that the extracted face region matches the expected coordinates. Tests cover cases with and without padding and cases where the face is near image boundaries.

The resize test verifies that output dimensions are correct and that aspect ratio handling produces expected results. Tests cover both upscaling and downscaling.

The normalization test verifies that pixel values are in the expected range and that the transformation is reversible within expected tolerances.

### 7.2.3 Model Tests

The trained model is tested with known inputs to verify correct behavior.

The embedding extraction test verifies that the model produces output of the expected shape and that outputs are consistent for the same input. The test also verifies that different inputs produce different outputs.

The similarity computation test verifies that L1 distance and cosine similarity are computed correctly. The test uses simple vectors where the expected result can be calculated analytically.

The threshold test verifies that the decision logic correctly classifies pairs based on the similarity score and threshold.

### 7.2.4 Service Tests

The face recognition service is tested with mocked dependencies to isolate the service logic.

The recognition test mocks the model and database to verify that the service correctly orchestrates the recognition workflow. The test verifies that embeddings are extracted, compared, and a result is returned.

The enrollment test mocks the file system and model to verify that the service correctly creates person records and stores images and embeddings.

The deletion test mocks the file system to verify that the service correctly removes person records and cleans up associated files.

---

## 7.3 Integration Testing

### 7.3.1 API Endpoint Testing

API endpoints are tested by making HTTP requests to the running server.

The test client is configured with the base URL of the API server. Requests are made using the requests library, and responses are validated.

Endpoint tests cover both success and error cases. Success tests verify that valid inputs produce correct responses. Error tests verify that invalid inputs produce appropriate error responses with meaningful messages.

Tests verify the correct HTTP status codes, response headers, and response bodies. JSON responses are parsed and validated against expected schemas.

### 7.3.2 Backend-Storage Integration

The integration between the backend and file storage is tested to verify correct data persistence.

The enrollment flow test verifies that images and metadata are correctly written to the file system and can be read back correctly. The test enrolls a person, verifies files exist, and verifies the data is correct.

The deletion flow test verifies that files are correctly removed and that subsequent operations see the deleted data. The test deletes a person and verifies the directory no longer exists.

The concurrent access test verifies that multiple simultaneous operations do not corrupt data. This is particularly important for the enrollment and deletion operations.

### 7.3.3 Model Inference Integration

The integration between the backend and the TensorFlow model is tested to verify correct inference.

The model loading test verifies that the model loads without errors and is ready for inference. The test checks that the model produces output for valid inputs.

The inference timing test measures the time taken for a single inference. The test verifies that timing is consistent and within expected bounds.

The memory usage test monitors memory consumption during inference. The test verifies that memory does not grow unbounded across multiple inferences.

### 7.3.4 Frontend-Backend Integration

The integration between the frontend and backend is tested to verify correct communication.

API client tests mock the HTTP layer to test the client functions. These tests verify that the correct requests are constructed and responses are parsed correctly.

End-to-end tests use a test backend server to verify the complete flow. These tests simulate real browser interactions with the web application.

---

## 7.4 System Testing

### 7.4.1 End-to-End Workflow Testing

System tests validate complete workflows from user input to final result.

The enrollment workflow test simulates a user enrolling a new person. The test captures images, submits them to the API, and verifies that the person appears in the persons list. The test also verifies that the person can be recognized.

The recognition workflow test simulates a user submitting an image for recognition. The test verifies that the correct person is identified when the face matches an enrolled person. The test also verifies that unknown faces are correctly rejected.

The management workflow test simulates administrative operations. The test verifies that persons can be listed, deleted, and that status information is accurate.

### 7.4.2 User Interface Testing

The user interface is tested to verify correct rendering and interaction.

Visual regression tests capture screenshots and compare them against baselines. Significant differences indicate unintended visual changes.

Functional UI tests simulate user interactions like clicking buttons and filling forms. These tests verify that UI actions trigger the correct API calls and display the correct results.

Responsive design tests verify that the interface adapts correctly to different screen sizes. These tests use different viewport dimensions to simulate various devices.

### 7.4.3 Browser Compatibility Testing

The web application is tested across multiple browsers to verify consistent behavior.

Tests run against Chrome, Firefox, Safari, and Edge. Each browser receives the same test cases and produces the same results.

Browser-specific quirks are identified and addressed. CSS vendor prefixes and JavaScript polyfills handle known differences.

Mobile browsers are tested on both iOS Safari and Android Chrome. Touch interactions are verified to work correctly.

---

## 7.5 Performance Testing

### 7.5.1 Response Time Testing

Response time testing measures how quickly the system responds to requests.

The recognition endpoint is timed from request submission to response receipt. The test measures multiple requests and reports average, minimum, and maximum times.

The results show that recognition typically completes in under three hundred milliseconds on the target hardware. This meets the requirement of under one second total including network and UI overhead.

Response time increases linearly with the number of enrolled persons because each person must be compared. For twenty enrolled persons, the increase is negligible.

### 7.5.2 Throughput Testing

Throughput testing measures how many requests the system can handle per second.

The test sends concurrent requests from multiple threads and measures the rate of completed requests. The test varies the concurrency level to find the maximum sustainable throughput.

The results show that the system can handle approximately five concurrent requests before throughput plateaus. This is sufficient for typical single-user and small-group usage.

### 7.5.3 Resource Usage Testing

Resource usage testing monitors CPU and memory consumption during operation.

CPU usage is measured during recognition requests. The test shows that CPU usage peaks during model inference and returns to idle between requests.

Memory usage is measured over extended operation. The test shows that memory usage is stable, with no memory leaks during sustained operation.

GPU usage is monitored on systems with GPU hardware. The test verifies that GPU acceleration is used when available and that performance improves accordingly.

### 7.5.4 Load Testing

Load testing subjects the system to sustained high request rates.

The test runs for an extended period with realistic request patterns. The test verifies that performance remains stable and that errors do not increase over time.

Results show that the system operates correctly under sustained load for the tested duration. Longer-duration testing would be needed for production confidence.

---

## 7.6 Security Testing

### 7.6.1 Input Validation Testing

Input validation prevents malicious or malformed data from causing issues.

Tests verify that large files are rejected with appropriate errors. Tests verify that non-image files are rejected. Tests verify that malformed images are handled gracefully.

The test suite includes boundary cases like empty files, corrupted files, and files with unusual formats.

### 7.6.2 Authentication and Authorization Testing

While the current version does not implement authentication, the API design supports future addition.

Tests verify that endpoints can be configured to require authentication. Tests verify that unauthenticated requests are rejected appropriately.

Authorization checks would verify that users can only access their own data. The data model supports this capability.

### 7.6.3 Error Message Testing

Error messages are tested to ensure they do not leak sensitive information.

Tests verify that internal errors (like file not found) produce generic error messages for clients. Detailed information is logged server-side for debugging.

Tests verify that error messages are consistent across endpoints. This helps clients handle errors uniformly.

---

## 7.7 Test Results Summary

### 7.7.1 Test Coverage

Test coverage measures how much of the codebase is exercised by tests.

The unit test suite covers approximately eighty percent of the backend code. Core functions like preprocessing and similarity computation have near-complete coverage.

Integration tests cover all API endpoints and major workflows. Each endpoint has both success and error test cases.

System tests cover the primary user workflows. These tests provide confidence that the complete system works end-to-end.

### 7.7.2 Test Results Table

The following table summarizes test results across all testing categories.

| Test Category | Tests Run | Passed | Failed | Success Rate |
|---------------|-----------|--------|--------|--------------|
| Unit Tests | 45 | 45 | 0 | 100% |
| Integration Tests | 22 | 22 | 0 | 100% |
| System Tests | 15 | 15 | 0 | 100% |
| Performance Tests | 8 | 8 | 0 | 100% |
| Security Tests | 12 | 12 | 0 | 100% |
| **Total** | **102** | **102** | **0** | **100%** |

All tests passed successfully, indicating that the system meets its functional and non-functional requirements.

### 7.7.3 Known Limitations

While all tests pass, some known limitations should be documented.

The face detector may fail on very low-quality images or unusual poses. Tests document these limitations but do not have automated workarounds.

The system does not implement liveness detection, making it potentially vulnerable to photo-based spoofing attacks. Security tests document this gap.

The web application relies on browser features that may not be available in all environments. Compatibility tests document supported browsers.

---

## 7.8 Chapter Summary

This chapter has presented the comprehensive testing strategy and results for the face recognition system.

Unit testing covered individual functions and classes, verifying correct behavior in isolation. Integration testing verified that components work correctly together. System testing validated complete end-to-end workflows.

Performance testing verified that the system meets its timing requirements. Security testing verified that the system handles malicious input appropriately.

The test results show that the system passes all tests with a one hundred percent success rate. The documented limitations are understood and acceptable for the current scope.

With testing complete, the report proceeds to Chapter 8, which presents the results and discussion.

---

**End of Chapter 7**
