# CHAPTER 6: IMPLEMENTATION

---

## 6.1 Introduction to Implementation

### 6.1.1 Implementation Overview

This chapter presents the implementation details of the face recognition system, translating the design specifications from Chapter 4 into working code. The implementation covers the backend API, the web frontend, the desktop application, and the supporting infrastructure.

Implementation follows the modular design established in the system design. Each component has clearly defined responsibilities and interfaces, enabling independent development and testing. The code is organized to reflect the logical structure of the system.

The implementation prioritizes correctness, readability, and maintainability over optimization. Where performance is critical, such as the inference pipeline, efficient algorithms are used, but the primary goal is correct behavior. The code is extensively commented to explain not just what happens but why.

### 6.1.2 Implementation Languages and Frameworks

The implementation uses multiple programming languages appropriate to each component.

Python serves as the primary language for the backend, machine learning components, and desktop application. Python's rich ecosystem provides libraries for web frameworks (FastAPI), machine learning (TensorFlow), image processing (OpenCV), and desktop UI (Kivy).

JavaScript powers the web frontend, built with the React library. React's component-based architecture matches the UI design, with each page and UI element implemented as a component.

HTML and CSS provide the structure and styling for web pages. CSS modules ensure styles are scoped to prevent conflicts.

Dockerfiles define the build process for containerized deployment, specifying the base images, dependencies, and build commands for each component.

### 6.1.3 Code Organization

The source code is organized in a directory structure that reflects the system architecture.

The backend code lives in the backend/app directory, with main.py as the entry point, service modules for business logic, and configuration files for settings.

The web frontend code lives in the frontend/src directory, organized with pages, components, and styles subdirectories.

The model code lives in the model/src directory, containing the model architecture, training scripts, and dataset handling.

The thesis code (including this document and generated diagrams) lives in the thesis directory.

---

## 6.2 Backend Implementation

### 6.2.1 FastAPI Application Structure

The FastAPI backend is organized into a main application module, service modules, and configuration.

The main module (main.py) creates the FastAPI application instance, configures middleware, and registers route handlers. It serves as the entry point that starts the API server.

The service modules encapsulate business logic. The face_system module handles model loading and inference. The service module provides higher-level operations like recognition and enrollment.

The configuration module loads settings from config.yaml and provides access throughout the application. This separation enables configuration changes without code modifications.

### 6.2.2 API Endpoint Implementation

The API endpoints are implemented as asynchronous functions decorated with FastAPI route decorators.

The root endpoint returns basic API information. The health endpoint checks model loading and data availability. These endpoints use the GET HTTP method and return JSON responses.

The recognize endpoint accepts file uploads. The function receives the uploaded file, saves it temporarily, processes it through the recognition pipeline, and returns the result. Error handling catches and reports any processing failures.

The persons endpoints provide CRUD operations. GET /persons lists all enrolled persons. POST /persons creates a new person with uploaded images. DELETE /persons/{person_id} removes a person and their data.

The settings endpoints manage configuration. GET /settings returns current settings. PUT /settings/threshold updates the recognition threshold.

### 6.2.3 Request and Response Models

Pydantic models define the structure of requests and responses, enabling automatic validation and documentation.

Request models define the expected fields and types for incoming data. The recognition request expects a file upload and optional threshold. The enrollment request expects a name and file.

Response models define the structure of outgoing data. The recognition response includes recognized status, person name, confidence score, and processing time. The persons response includes the list of enrolled persons.

FastAPI automatically validates incoming data against these models, returning appropriate error responses for invalid input. The models also generate OpenAPI documentation that describes the expected formats.

### 6.2.4 CORS Configuration

Cross-Origin Resource Sharing (CORS) configuration enables the web frontend to communicate with the backend from different origins.

The FastAPI application adds CORSMiddleware with permissive settings for development. The allow_origins parameter is set to asterisk to permit requests from any origin. The allow_methods and allow_headers parameters permit all HTTP methods and headers.

In production, the origins would be restricted to the specific frontend domain. This prevents unauthorized access while enabling legitimate cross-origin requests.

### 6.2.5 Error Handling

The backend implements consistent error handling that returns meaningful error responses to clients.

HTTPException is raised for expected error conditions like invalid input or missing resources. The exception includes a status code and detail message.

Try-except blocks catch unexpected exceptions in route handlers. Caught exceptions are logged for debugging and converted to HTTP 500 responses for the client.

The error responses include a clear message explaining what went wrong. This helps clients provide useful feedback to users and debug their own code.

---

## 6.3 Face Recognition Service Implementation

### 6.3.1 Service Class Structure

The FaceRecognitionSystem class encapsulates all machine learning functionality. This class is instantiated once at application startup and reused for all requests.

The constructor loads the trained model and initializes the face detector. These expensive operations happen once rather than for each request, improving throughput.

The service provides methods for each major operation: loading models, switching models, preprocessing images, extracting embeddings, and recognizing faces.

Thread safety is achieved through lazy loading and model caching. The model is loaded once and shared across requests, with appropriate synchronization for any state changes.

### 6.3.2 Model Loading

Model loading happens at service initialization and when switching between models.

The load_model method handles loading from the HDF5 file. It uses TensorFlow's load_model function with custom objects registered for the custom layers.

If loading fails, the service logs the error and raises an exception. The application cannot function without a loaded model, so startup fails clearly rather than silently degrading.

The service also supports loading different model variants. This enables experimentation with alternative architectures without code changes.

### 6.3.3 Image Preprocessing

The preprocessing methods transform raw images into the format expected by the model.

The preprocess_image method accepts an image path and returns a preprocessed numpy array. It calls the face detector to locate faces, then crops, resizes, and normalizes the detected face.

The method returns None if no face is detected, enabling callers to handle this case appropriately. The method also tracks whether a face was detected for logging purposes.

The preprocessing is stateless; given the same input, it always produces the same output. This enables consistent results and simplifies testing.

### 6.3.4 Embedding Extraction

Embedding extraction runs the preprocessed image through the model to obtain the feature vector.

The get_embedding method takes a preprocessed image array and returns the embedding vector. The array is first expanded to add a batch dimension, then passed to the model's predict method.

The method returns None if the embedding cannot be computed. This would occur if the model is not loaded or the input is invalid.

Embeddings are numpy arrays with shape (1, 256) for single images. The arrays can be stored and compared efficiently.

### 6.3.5 Recognition Logic

The recognize method implements the complete recognition logic from image path to identity result.

The method first extracts the embedding for the query image. If extraction fails, it returns an error result.

The method then loads embeddings for all enrolled persons. For each person, it computes the cosine similarity between the query embedding and the person's stored embedding.

The person with the highest similarity is identified. If the highest similarity exceeds the configured threshold, the person is recognized. Otherwise, the result is unknown.

The method returns a tuple of (person_name, similarity_score). The caller formats this into the appropriate response.

### 6.3.6 Person Management

Person management methods handle enrollment and deletion operations.

The add_person method creates a new person directory, saves images, and extracts embeddings. It returns a status indicating success or failure with details.

The delete_person method removes a person's directory and all associated data. It returns a status indicating success or failure.

The list_persons method returns information about all enrolled persons. It scans the persons directory and collects metadata for each person.

---

## 6.4 Web Frontend Implementation

### 6.4.1 React Application Structure

The React application is organized into pages, components, and services.

The pages directory contains top-level page components: RecognizePage, EnrollPage, PersonsPage, and SettingsPage. Each page corresponds to a route in the application.

The components directory contains reusable UI components like WebcamCapture, ImageUploader, and ResultDisplay. These components are composed to build the pages.

The services directory contains API client functions that communicate with the backend. The API service abstracts the HTTP requests and response handling.

### 6.4.2 State Management

React's useState and useContext hooks manage application state.

The App component maintains the current page and settings in state. The state is passed down to child components through props and context.

The API context provides access to backend configuration and functions. Components consume this context rather than directly calling the API.

Local state manages component-specific data like form inputs and image previews. This state is ephemeral and reset when the component unmounts.

### 6.4.3 Recognition Page Implementation

The RecognizePage enables users to submit images for verification.

The page uses the WebcamCapture component for camera-based input. The component accesses the device camera, displays the video stream, and captures frames on user command.

The page also supports file upload through the ImageUploader component. The component accepts image files and displays a preview.

After capture or upload, the page sends the image to the backend /recognize endpoint. The response is displayed using the ResultDisplay component, showing whether the face was recognized and the person's name if so.

### 6.4.4 Enrollment Page Implementation

The EnrollPage guides users through adding new persons.

The page includes a name input field where users enter the person's name. Validation ensures the name is non-empty and unique.

The page includes an image capture area supporting both camera and upload modes. Users capture or upload multiple images, with the page showing a thumbnail for each captured image.

A counter shows how many images have been captured and highlights when the recommended minimum is reached.

When the user clicks Enroll, the page sends the name and images to the backend /persons endpoint. Success shows a confirmation message, and the page resets for another enrollment.

### 6.4.5 Persons Page Implementation

The PersonsPage displays enrolled persons and enables management.

The page fetches the list of persons from the backend /persons endpoint on mount. The list is displayed in a grid of person cards.

Each card shows the person's name and image count. A delete button triggers confirmation and deletion through the backend DELETE /persons/{id} endpoint.

After deletion, the page refreshes the list to reflect the change. Optimistic updates could be implemented but were not included in this version.

### 6.4.6 Settings Page Implementation

The SettingsPage provides configuration options.

The recognition threshold is displayed with a slider control. Moving the slider updates the threshold through the backend PUT /settings/threshold endpoint.

System information shows the current model, number of enrolled persons, and total images. This information helps users understand the system state.

Links to API documentation and project resources are provided for users who want more information.

### 6.4.7 Styling and Theming

The application uses CSS modules for scoped styling. Each component has an associated CSS file with styles specific to that component.

CSS variables define the color palette and spacing, enabling consistent theming. The dark theme uses dark backgrounds with light text and accent colors.

Responsive design ensures the application works on different screen sizes. Flexbox and grid layouts adapt to available space.

---

## 6.5 Desktop Application Implementation

### 6.5.1 Kivy Application Structure

The Kivy desktop application is a Python program that creates a graphical user interface.

The application is organized into a main App class and custom Widget classes. The App class initializes the application and defines the root widget. The Widget classes implement specific UI elements.

The Builder loads the KV language file that defines the widget hierarchy and styling. This separation of code and layout improves readability.

### 6.5.2 Camera Integration

Camera integration uses OpenCV to access the webcam and display the video feed.

The application creates a VideoCapture object to access the default camera. A Clock schedules periodic frame updates at approximately thirty frames per second.

Each frame is read from the camera and converted from OpenCV's BGR format to RGB for display. The frame is converted to a Kivy texture and displayed in an Image widget.

When the user captures, the current frame is stored for processing. The captured frame is saved to a temporary file for submission to the backend.

### 6.5.3 UI Layout

The desktop UI follows a layout with camera preview, controls, and results areas.

The camera preview occupies the left portion of the window, showing the live webcam feed. Below the preview, buttons enable capture and mode switching.

The right portion shows results and status information. Recognition results are displayed prominently when available. The current threshold setting is shown with an adjustable slider.

The layout uses BoxLayout and GridLayout widgets to arrange components. The layouts are nested to achieve the desired structure.

### 6.5.4 Backend Communication

The desktop application communicates with the backend through HTTP requests, the same as the web application.

The requests library provides HTTP client functionality. The application sends multipart form requests for recognition and enrollment.

The same API endpoints are used, ensuring consistent behavior between web and desktop interfaces. Any changes to the API are automatically reflected in both interfaces.

---

## 6.6 Infrastructure Implementation

### 6.6.1 Docker Configuration

Docker containers package each component with its dependencies for consistent deployment.

The backend Dockerfile starts from a Python base image, copies the application code, installs dependencies from requirements.txt, and sets the command to run the API server.

The frontend Dockerfile starts from a Node base image, copies the application code, installs dependencies from package.json, builds the production bundle, and serves it with a simple HTTP server.

Multi-stage builds reduce the final image size by separating build-time and runtime dependencies.

### 6.6.2 Docker Compose Configuration

Docker Compose orchestrates multi-container deployments.

The compose file defines services for backend and frontend. Each service specifies its build context, port mapping, volume mounts, and environment variables.

The backend service exposes port eight thousand for API access. The frontend service exposes port three thousand for web access.

Named volumes persist data across container restarts. Environment variables configure runtime behavior like logging levels.

### 6.6.3 Environment Configuration

Environment variables configure the application for different deployment scenarios.

The TF_CPP_MIN_LOG_LEVEL variable controls TensorFlow logging. Setting it to two or three reduces log verbosity in production.

The VITE_API_URL variable tells the frontend where to find the backend. This enables deploying frontend and backend on different hosts.

The PYTHONUNBUFFERED variable ensures Python output is not buffered, which is important for seeing logs in container environments.

---

## 6.7 Key Algorithms and Functions

### 6.7.1 Face Detection Algorithm

The face detection function locates faces in images using Haar cascades.

The implementation loads the cascade classifier once and reuses it for all detections. The classifier is loaded from the OpenCV data directory.

The detectMultiScale function performs multi-scale detection. It returns rectangles for all detected faces, sorted by confidence.

The function filters detections by size, removing faces that are too small relative to the image. This filters out false positives that are typically small.

### 6.7.2 Image Preprocessing Algorithm

The preprocessing function transforms images for model input.

The function first reads the image using OpenCV. If reading fails, it returns None.

The function then applies face detection to locate faces. If no face is found, it returns None.

The detected face is cropped with padding and resized to ninety-six by ninety-six pixels. The image is converted from BGR to RGB and normalized to negative one to positive one.

The preprocessed image is returned as a numpy array with shape ninety-six by ninety-six by three and dtype float32.

### 6.7.3 Cosine Similarity Algorithm

Cosine similarity measures the angle between vectors.

The implementation normalizes both vectors to unit length, computes their dot product, and returns the result.

Mathematically, cosine similarity equals the sum of element-wise products divided by the product of vector norms. When vectors are unit-normalized, the denominator is one, simplifying to just the dot product.

The implementation uses numpy's linalg module for vector operations. The result is a scalar between negative one and one.

### 6.7.4 Person Slugification Algorithm

Slugification converts display names to directory-safe identifiers.

The algorithm converts to lowercase, replaces spaces with underscores, and removes non-alphanumeric characters except underscores.

The implementation uses regular expressions to identify and filter characters. This ensures only valid filename characters remain.

The slug is checked for uniqueness before use. If a slug already exists, an error is returned to the user.

---

## 6.8 Integration Points

### 6.8.1 Frontend-Backend Integration

The frontend communicates with the backend through HTTP requests.

The fetch API (or axios library) sends requests to the backend API endpoints. Requests include appropriate headers for JSON content type.

Request bodies are formatted according to the endpoint specification. File uploads use multipart form encoding. Simple parameters use URL query strings.

Responses are parsed as JSON and validated against expected structures. Errors are caught and displayed to the user.

### 6.8.2 Backend-Model Integration

The backend loads and uses the trained model through TensorFlow's Keras API.

Model loading happens once at startup. The load_model function reads the HDF5 file and reconstructs the model architecture and weights.

Inference uses the predict method, which runs the model on input data and returns predictions. The method is called with preprocessed images and returns similarity scores.

Memory management ensures the model stays in memory between requests. Model tensors are cleared after use to free GPU memory if applicable.

### 6.8.3 Data Storage Integration

The backend accesses persistent storage through the file system.

The pathlib module provides cross-platform path manipulation. Paths are constructed by joining directory and filename components.

The json module reads and writes metadata files. The OpenCV imwrite function saves images in JPEG format.

File operations are wrapped in try-except blocks to handle permission errors and disk full conditions gracefully.

---

## 6.9 Chapter Summary

This chapter has presented the implementation details of the face recognition system.

The backend implementation covers the FastAPI application structure, API endpoints, request/response models, and error handling. The face recognition service encapsulates all ML functionality.

The frontend implementation covers the React application structure, state management, page implementations, and styling. Each page provides specific functionality for users.

The desktop implementation covers the Kivy application structure, camera integration, UI layout, and backend communication. The desktop app provides equivalent functionality to the web app.

The infrastructure implementation covers Docker configuration, Docker Compose orchestration, and environment configuration. These enable containerized deployment.

With the implementation documented, the report proceeds to Chapter 7, which covers the testing strategy and results.

---

**End of Chapter 6**
