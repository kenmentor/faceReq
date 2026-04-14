# CHAPTER 4: SYSTEM DESIGN

---

## 4.1 Introduction to System Design

### 4.1.1 Purpose of System Design

This chapter presents the detailed design of the face recognition system, translating the methodology described in the previous chapter into concrete specifications that guide implementation. The system design encompasses the overall architecture, component interactions, data structures, and interface specifications that define how the system is built.

Good system design is essential for building maintainable, scalable, and reliable software. The design phase identifies potential issues before code is written, enabling solutions that would be expensive or impossible to implement later. The design also provides documentation that helps developers understand the system and make consistent decisions during implementation.

The design presented here balances several competing concerns. Performance must be sufficient for real-time operation on target hardware. Complexity must be manageable for a single-developer project with limited time. Maintainability must enable future extensions and bug fixes. Portability must support deployment on different platforms and configurations.

### 4.1.2 Design Principles

Several principles guided the system design decisions, ensuring consistency and coherence across components.

The first principle is separation of concerns, which means organizing code into distinct modules with clear responsibilities. The face detection module does not know about network inference, and the API server does not know about image preprocessing. This separation simplifies testing and enables component reuse.

The second principle is configuration over convention, meaning behavior is controlled through configuration files rather than hard-coded assumptions. The location of data directories, the model file path, and the recognition threshold are all configurable. This enables deployment in different environments without code changes.

The third principle is graceful degradation, which means the system continues operating when possible even when components fail. If the webcam is unavailable, the system can still process uploaded images. If face detection fails, the system returns an informative error rather than crashing.

The fourth principle is extensibility, which means the architecture supports adding new features without major refactoring. New API endpoints can be added without modifying existing endpoints. New model architectures can be integrated by implementing a common interface.

### 4.1.3 Design Documents Overview

The system design is documented through multiple artifacts that together provide a complete picture of the system.

The architecture diagram shows the high-level organization of components and their relationships. It answers questions about what components exist and how they interact.

The data flow diagrams show how data moves through the system from input to output. They answer questions about what transformations are applied and in what order.

The entity relationship diagram shows the data structures that persist beyond individual requests. It answers questions about what data is stored and how it is organized.

The interface specifications show how components interact through APIs and function calls. They answer questions about what operations are available and how to use them.

---

## 4.2 High-Level System Architecture

### 4.2.1 Three-Tier Architecture

The face recognition system follows a three-tier architecture that separates presentation, processing, and data concerns into distinct layers. This separation enables independent development, testing, and deployment of each layer.

The presentation tier encompasses all user-facing components that handle interaction with humans. This includes the React-based web application that runs in browsers, the Kivy-based desktop application that runs as a native program, and the FastAPI-generated API documentation that helps developers understand the API. The presentation tier receives user input, sends requests to the processing tier, and displays results.

The processing tier contains the core business logic that transforms inputs into outputs. This includes the FastAPI backend that receives HTTP requests and returns responses, the face recognition service that handles model loading and inference, and the image preprocessing functions that prepare images for the network. The processing tier does not know anything about user interfaces; it simply processes requests according to defined rules.

The data tier manages persistent storage of models, images, and metadata. This includes the trained neural network model stored as an HDF5 file, the enrolled person images stored in a directory structure, and the configuration stored as YAML and JSON files. The data tier does not perform any processing; it only stores and retrieves data.

Communication between tiers follows well-defined interfaces. The presentation tier communicates with the processing tier through HTTP requests. The processing tier communicates with the data tier through file system operations. These interfaces are stable and enable changes within each tier without affecting others.

### 4.2.2 Component Diagram

The system consists of several major components that work together to provide face recognition functionality.

The web application component provides the browser-based user interface. It is built with React and includes pages for recognition, enrollment, person management, and settings. The web application runs entirely in the browser and communicates with the backend through REST API calls.

The desktop application component provides a standalone user interface built with Kivy. It includes similar functionality to the web application but runs as a native Python application without requiring a browser. The desktop application also communicates with the backend through REST API calls.

The API server component provides the backend service that handles all recognition operations. It is built with FastAPI and exposes endpoints for recognition, enrollment, person management, and system configuration. The API server loads the trained model at startup and keeps it in memory for fast inference.

The face recognition service component encapsulates the core machine learning logic. It handles model loading, image preprocessing, embedding extraction, and similarity computation. The API server delegates to this service for all ML-related operations.

The data management component handles storage and retrieval of persistent data. It manages the directory structure for enrolled persons, the model file, and configuration files. This component provides a consistent interface for data access that abstracts the underlying file system structure.

### 4.2.3 Deployment Architecture

The system can be deployed in several configurations depending on requirements and available infrastructure.

The single-machine deployment runs all components on a single computer. The backend API server runs as a Python process, the web application is served by a development server, and the model files are stored locally. This configuration is suitable for development, testing, and small-scale deployment.

The containerized deployment packages each component in a Docker container. The backend container runs the API server, the frontend container runs the web application, and an nginx container provides reverse proxy and load balancing. This configuration is suitable for production deployment on servers or in cloud environments.

The distributed deployment separates components across multiple machines. The API servers run on dedicated backend machines, the web application is served from a CDN, and the model files are stored on networked storage. This configuration provides scalability for high-demand scenarios.

---

## 4.3 Data Flow Design

### 4.3.1 Recognition Data Flow

The recognition data flow begins when a user submits an image for verification and ends when the system returns an identity decision. Understanding this flow is essential for implementing and debugging the system.

The flow starts at the presentation tier when the user captures an image through webcam or uploads an image file. The presentation component validates that an image was actually provided and that it has an acceptable format (JPEG, PNG). The image is encoded as multipart form data in an HTTP POST request to the recognition endpoint.

The HTTP request travels over the network to the API server, which is listening on the configured port. FastAPI's request handling extracts the uploaded file from the multipart form and saves it to a temporary file. The temporary file path is passed to the face recognition service.

The face recognition service loads the image from the temporary file using OpenCV. The image is converted from its stored format to a numpy array representation. Face detection locates any faces in the image, and the preprocessing pipeline extracts, crops, resizes, and normalizes the detected face.

The preprocessed image is passed to the neural network for embedding extraction. TensorFlow's predict function runs the image through the model and returns the embedding vector. The embedding is compared against stored embeddings for enrolled persons.

For each enrolled person, the service computes cosine similarity between the query embedding and the person's stored embedding. The person with the highest similarity is identified. If the highest similarity exceeds the configured threshold, the person is marked as recognized; otherwise, the result is unknown.

The service returns the recognition result to the API server, which formats it as a JSON response. The JSON response includes the recognized status, person name if recognized, similarity score, configured threshold, processing time, and any error messages. This response travels back to the presentation tier for display to the user.

### 4.3.2 Enrollment Data Flow

The enrollment data flow adds new persons to the system database. This flow is more complex than recognition because it must handle multiple images and create persistent storage.

The flow begins when a user provides a name and one or more facial images. The name must be non-empty and unique among existing enrolled persons. The images must show clear, single faces. The presentation component validates these constraints before sending to the backend.

The API server receives the name and images and creates a directory for the new person. The directory name is derived from the name using a slugification process that converts spaces to underscores, removes special characters, and lowercases all letters. For example, "John Doe" becomes "john_doe".

For each submitted image, the server runs the full recognition pipeline through embedding extraction. The embedding is stored in memory temporarily. If any image fails to process (no face detected, face too small, and so on), enrollment fails with an error message indicating which image had problems.

After all images are successfully processed, the server saves each image to the person's directory with sequential numeric names. The server also writes a metadata file that includes the person's display name, enrollment timestamp, image count, and individual embeddings.

Finally, the server returns a success response indicating the person was enrolled with how many images were stored. The person immediately becomes available for recognition without requiring any model retraining or server restart.

### 4.3.3 Person Deletion Data Flow

The deletion data flow removes enrolled persons from the system. This is the inverse of enrollment and must handle cleanup of all associated data.

The flow begins when a user requests deletion of a specific person by their identifier. The identifier is the slugified name used as the directory name. The API server validates that the person exists before proceeding.

If the person exists, the server deletes the person's directory and all its contents. This removes all enrolled images and embeddings. The deletion is permanent; there is no trash or recovery mechanism.

The server returns a success response confirming deletion. If the person did not exist, the server returns a not found error. The recognition pipeline immediately stops recognizing the deleted person, as their embeddings are no longer in storage.

### 4.3.4 Data Flow Diagram Description

The data flow diagrams in the accompanying figures illustrate the movement of data through the system at different levels of abstraction.

The Level 0 diagram shows the system as a single process with external entities representing users and the file system. This context diagram establishes the boundary between the system and its environment without exposing internal details.

The Level 1 diagram decomposes the system into major processes including receive input, process image, verify identity, store data, and return result. Data stores represent the persistent storage of models, persons, and configuration. Data flows show how information moves between processes and stores.

---

## 4.4 Database and Storage Design

### 4.4.1 Storage Strategy

The system uses a file-based storage strategy rather than a traditional database management system. This choice simplifies deployment by eliminating database dependencies while providing sufficient capability for the expected data volumes.

File-based storage stores data as regular files in the file system, organized according to a defined directory structure. Person images are stored as JPEG files in directories named after each person. Metadata is stored as JSON files that can be read and written by Python's json module. Configuration is stored as YAML files readable by PyYAML.

The advantages of file-based storage include simplicity, portability, and transparency. There is no database server to install, configure, or manage. Files can be backed up with standard tools and inspected with normal file utilities. The storage is portable across platforms without conversion.

The disadvantages include limited query capability and potential scalability issues. Finding all persons with more than three images requires scanning all metadata files. Adding thousands of persons may slow directory operations. For this project's scale, these limitations are acceptable.

### 4.4.2 Directory Structure

The application stores all data in a structured hierarchy of directories. The root directory is named application_data and contains subdirectories for different types of data.

The persons directory contains one subdirectory for each enrolled person. Each person's subdirectory contains their enrolled images and metadata file. This structure makes it easy to list all persons and to access a specific person's data.

The input_images directory holds images submitted for verification that are not yet associated with a person. These images may be processed and then discarded. Keeping them separate prevents confusion with enrolled images.

The verification_images directory contains reference images used for verification in certain configurations. These images serve as comparison points when processing input images.

The model directory contains the trained neural network file. The model file is typically named trained_model.h5 and contains both the architecture and weights. Separating the model from other data simplifies model updates.

### 4.4.3 Entity Relationship Diagram

The entity relationship diagram shows the logical structure of the data stored by the system.

The Person entity represents an enrolled individual. Attributes include the person identifier (slugified name), display name, enrollment timestamp, and image count. The identifier is the primary key that uniquely identifies each person.

The Image entity represents a facial image associated with a person. Attributes include the image filename, the embedding vector, and storage metadata. Each image belongs to exactly one person, establishing a one-to-many relationship.

The VerificationLog entity records verification attempts for audit and analytics purposes. Attributes include the timestamp, query image reference, result (recognized or not), matched person if any, and similarity score. This entity enables tracking system usage over time.

### 4.4.4 Metadata Schema

Each enrolled person has an associated metadata file in JSON format. The metadata captures information about the person that is not stored in the image filenames.

The metadata schema includes the person identifier as the filename key, the display name for human-readable identification, the enrollment timestamp in ISO format, the image count indicating how many images are enrolled, and the average embedding computed across all enrolled images.

The average embedding provides a compact representation of the person's face. Rather than comparing against all individual embeddings, verification can compare against just the average, improving performance with minimal accuracy loss.

---

## 4.5 API Design

### 4.5.1 REST API Overview

The system exposes a REST API that enables programmatic access to all functionality. The API follows REST conventions with appropriate HTTP methods, status codes, and content types.

The API is built with FastAPI, which provides automatic request validation, response serialization, and API documentation. The documentation is accessible at the /docs endpoint using Swagger UI.

All API endpoints are prefixed with the application root. For a server running on localhost port 8000, the full endpoint URLs would be like http://localhost:8000/recognize or http://localhost:8000/persons.

The API accepts requests with JSON bodies for simple parameters and multipart form data for file uploads. Responses are always JSON format. Error responses include an error field with a human-readable message explaining what went wrong.

### 4.5.2 API Endpoints

The API defines several endpoints that together provide complete functionality.

The root endpoint at GET / returns basic information about the API including version and status. This endpoint is useful for verifying that the API is running.

The health endpoint at GET /health returns detailed system status including whether the model is loaded and how many persons are enrolled. This endpoint is useful for monitoring and load balancer health checks.

The settings endpoint at GET /settings returns current configuration including the recognition threshold, current model, and system statistics. The PUT /settings/threshold endpoint updates the recognition threshold.

The recognition endpoint at POST /recognize accepts an image file and optional threshold parameter. It returns the recognition result including whether the face was recognized, the person's name, the similarity score, and processing time.

The persons endpoints provide CRUD operations for enrolled persons. GET /persons returns a list of all enrolled persons with their metadata. POST /persons creates a new person with uploaded images. GET /persons/{person_id} returns details for a specific person. DELETE /persons/{person_id} removes a person.

### 4.5.3 Request and Response Formats

Each API endpoint has specific request and response formats that clients must follow.

The recognition endpoint expects a multipart form upload with an image file field named "file" and an optional numeric field named "threshold". The response includes recognized (boolean), person_name (string or null), confidence (float), threshold (float), message (string), and processing_time (float).

The create person endpoint expects a multipart form with a "name" field and a "file" field for the image. The response includes status (string), message (string), and person object with id, name, and image_count.

The list persons response includes status (string), message (string), and persons array containing objects with id, name, and image_count for each enrolled person.

The update threshold endpoint expects a form field named "threshold" with a float value between zero and one. The response confirms the new threshold value.

### 4.5.4 Error Handling

The API implements consistent error handling that helps clients understand and handle error conditions.

HTTP status codes indicate the general category of error. Four hundred status codes indicate client errors (bad request, not found), while five hundred status codes indicate server errors.

The response body includes an error message explaining what went wrong in human-readable form. For example, if no face is detected, the message might be "No face detected in the uploaded image. Please ensure the image shows a clear, frontal view of a face."

Validation errors include details about which fields failed validation and what constraints were violated. This helps API clients provide better user feedback about input problems.

---

## 4.6 User Interface Design

### 4.6.1 Web Application Layout

The web application provides a modern, responsive interface built with React. The layout consists of a sidebar for navigation and a main content area that changes based on the selected page.

The sidebar contains navigation links to all pages: Recognize, Enroll, Persons, and Settings. The current page is highlighted to show which section the user is viewing. The sidebar also shows system status information such as whether the backend is connected.

The main content area displays the selected page's content. Each page has a distinct purpose and layout tailored to its function. The layout is responsive, adapting to different screen sizes for use on both desktop and mobile devices.

The header shows the application title and any global actions or notifications. The footer contains version information and links to help documentation.

### 4.6.2 Recognition Page Design

The Recognize page enables users to submit images for verification against enrolled persons. The page has two modes: webcam capture and file upload.

In webcam mode, the page displays a live video feed from the device camera. A capture button triggers snapshot of the current frame. The captured image is displayed in a preview area. A verify button submits the image for recognition.

In upload mode, the page displays a file input that accepts image files. Selecting a file shows the image preview. A verify button submits the selected file for recognition.

After verification, the result is displayed prominently. If recognized, the person's name and confidence score are shown with a success indicator. If not recognized, an unknown indicator is shown with the best match score even though it was below threshold.

### 4.6.3 Enrollment Page Design

The Enroll page enables users to add new persons to the system. The page guides users through the enrollment process with clear instructions and feedback.

The page begins with a name input field where the user enters the person's name. The name is validated to ensure it is non-empty and unique among existing persons.

Below the name input, the page shows an image capture area. Users can use the webcam to capture multiple photos or upload files. The page displays thumbnails of captured images with options to remove any unwanted photos.

The page shows a progress indicator indicating how many images have been captured. The recommended minimum of three images is highlighted. Users can enroll with fewer images but accuracy may be reduced.

When ready, the user clicks the Enroll button to submit all data to the backend. The page shows a loading indicator during processing. Upon success, a confirmation message appears and the page can be reset for another enrollment.

### 4.6.4 Persons Page Design

The Persons page displays all enrolled persons and enables management operations. The page shows a grid of person cards, each representing one enrolled individual.

Each person card shows the person's display name and the number of enrolled images. The card may show a thumbnail of one enrolled image if available. The card includes a delete button to remove the person.

Clicking a person card shows detailed information about that person including all enrolled images. From this detail view, users can add additional images or delete existing ones.

The page includes a search or filter function to find specific persons by name. This is useful when many persons are enrolled.

### 4.6.5 Settings Page Design

The Settings page provides configuration options for the system. The page displays current settings and enables modification.

The primary setting is the recognition threshold. A slider allows adjustment from zero to one with the current value displayed. Changes take effect immediately for subsequent recognitions.

The page shows system information including the loaded model name, the total number of enrolled persons, and the total number of enrolled images. This information helps users understand the system state.

The page includes links to API documentation and the project's source code repository for users who want to understand or modify the system.

---

## 4.7 Technology Stack

### 4.7.1 Machine Learning Stack

The machine learning components use TensorFlow as the primary framework. TensorFlow provides comprehensive support for building, training, and deploying neural network models. The Keras API within TensorFlow simplifies model construction through a high-level interface.

TensorFlow was chosen for its maturity, documentation, and ecosystem. It has been in development since 2015 and has extensive community support. Pre-trained models and tutorials are readily available. The TensorFlow ecosystem includes TensorFlow Lite for mobile deployment and TensorFlow.js for browser-based inference.

OpenCV provides computer vision functionality including image reading, writing, and manipulation. The cv2 module interfaces with OpenCV's C++ libraries. OpenCV includes the Haar cascade classifiers used for face detection.

NumPy provides efficient numerical operations on arrays. The preprocessing pipeline uses NumPy for image array manipulation. NumPy's broadcasting and vectorization enable fast computation without explicit loops.

### 4.7.2 Backend Stack

The backend API is built with FastAPI, a modern Python web framework. FastAPI provides automatic request validation using Pydantic models, automatic response serialization, and automatic API documentation generation.

FastAPI was chosen for its performance, developer experience, and modern design. It uses Starlette for the web layer and Pydantic for data validation. Async request handling enables high throughput for I/O-bound operations.

Uvicorn serves as the ASGI server that runs the FastAPI application. Uvicorn implements the ASGI specification for Python async web applications. It supports multiple workers for production deployment.

Pydantic provides data validation and settings management using Python type annotations. Request and response models are defined as Pydantic classes that automatically validate incoming data and serialize outgoing data.

### 4.7.3 Frontend Stack

The web application is built with React, a JavaScript library for building user interfaces. React uses a component-based architecture where UIs are composed from reusable components.

React was chosen for its component model, ecosystem, and widespread adoption. The virtual DOM provides good performance, and the extensive library ecosystem covers most UI needs.

Vite serves as the build tool and development server for the React application. Vite provides fast startup and hot module replacement during development. In production, Vite bundles the application for optimal loading.

CSS modules provide styling with local scope to prevent conflicts. A CSS variables-based design system enables consistent theming across components.

### 4.7.4 Desktop Stack

The desktop application is built with Kivy, an open-source Python framework for developing cross-platform applications. Kivy provides a responsive UI toolkit with touch support.

Kivy was chosen for its cross-platform capability and Python integration. The same codebase runs on Windows, macOS, and Linux. Kivy's Python foundation simplifies integration with the backend's machine learning components.

### 4.7.5 Infrastructure Stack

Docker containerizes the application components for consistent deployment. Docker images package the application code along with all dependencies, ensuring identical behavior across environments.

Docker Compose orchestrates multi-container deployments. The compose file defines services for the backend, frontend, and optionally nginx. Volumes persist data across container restarts.

Git provides version control for source code. GitHub hosts the remote repository and provides collaboration features including issues and pull requests.

---

## 4.8 Configuration Management

### 4.8.1 Configuration Files

The system uses configuration files to customize behavior without code changes. Different configuration files control different aspects of the system.

The config.yaml file contains application-level settings including server host and port, paths to data directories, and default thresholds. This file is read at application startup and provides sensible defaults for all environments.

The backend/requirements.txt file lists Python package dependencies with version constraints. Installing these packages with pip ensures all required libraries are available.

The frontend/package.json file lists JavaScript dependencies and defines scripts for development and production builds. The npm install command installs all dependencies.

The docker-compose.yml file defines how containers are built and run. It specifies build contexts, port mappings, volume mounts, and environment variables.

### 4.8.2 Environment Variables

Environment variables provide another mechanism for configuration, particularly for deployment-specific values.

The TF_CPP_MIN_LOG_LEVEL variable controls TensorFlow logging verbosity. Setting it to two suppresses most log messages, leaving only errors visible.

The PYTHONUNBUFFERED variable ensures Python output is not buffered, which is important for seeing logs in container environments.

The VITE_API_URL variable tells the frontend where to find the backend API. This enables the frontend to be deployed independently of the backend.

### 4.8.3 Runtime Configuration

Some configuration can be modified at runtime through the API or user interface without restarting the application.

The recognition threshold is stored in a JSON configuration file and loaded at startup. The settings API endpoint allows reading the current value, and the threshold update endpoint allows changing it. The new value takes effect immediately for subsequent requests.

Model selection could similarly be made configurable at runtime by loading different model files based on configuration. This feature is partially implemented but not fully integrated into the UI.

---

## 4.9 Chapter Summary

This chapter has presented the detailed design of the face recognition system, covering the architecture, data flows, storage, interfaces, and technology choices.

The high-level architecture section described the three-tier design separating presentation, processing, and data concerns. The component diagram showed the major system components and their relationships.

The data flow section traced the path of data through the system for recognition, enrollment, and deletion operations. Understanding these flows is essential for implementing and debugging the system.

The storage section explained the file-based approach to data persistence including directory structure, entity relationships, and metadata formats.

The API section documented the REST endpoints, request/response formats, and error handling conventions that enable programmatic access.

The UI section described the design of web application pages including recognition, enrollment, persons management, and settings.

The technology stack section justified the choice of tools and frameworks used to build each component of the system.

With the design documented, the report proceeds to Chapter 5, which covers the implementation details including code structure, key algorithms, and integration points.

---

**End of Chapter 4**
