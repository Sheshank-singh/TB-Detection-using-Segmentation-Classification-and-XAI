# 🫁 TB Chest X-Ray Classifier Web Application

## Project Overview

This project is a web-based application designed to assist in the classification of Tuberculosis (TB) from chest X-ray images. It leverages deep learning models for both lung segmentation and TB classification, providing visual explanations through Grad-CAM heatmaps. The application features a user-friendly interface for uploading X-ray images and instantly viewing the prediction results along with insightful visualizations.

## Features

*   **Image Upload:** Easily upload chest X-ray images via drag-and-drop or file selection.
*   **TB Classification:** Classifies X-ray images as "Normal" or "TB" with a confidence score.
*   **Lung Segmentation:** Visualizes the segmented lung regions from the uploaded X-ray.
*   **Grad-CAM Heatmaps:** Generates and displays Grad-CAM heatmaps to highlight areas of the X-ray that are most influential in the model's classification decision.
*   **Modern User Interface:** A clean, responsive, and intuitive design built with React.js.

## Technologies Used

### Backend (Python - FastAPI)

*   **Python 3.x:** The core programming language.
*   **FastAPI:** A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.
*   **TensorFlow/Keras:** For loading and running the deep learning models (EfficientNetB0 for classification, U-Net-like for segmentation).
*   **OpenCV (cv2):** Used for image processing tasks such as resizing, color space conversions, and image encoding/decoding.
*   **NumPy:** Fundamental package for numerical computation in Python.
*   **scikit-image (skimage):** For advanced image processing, specifically `measure.label` and `measure.regionprops` for mask post-processing.
*   **SciPy (scipy.ndimage):** Used for `binary_fill_holes` in mask post-processing.
*   **Uvicorn:** An ASGI server for running the FastAPI application.

### Frontend (React.js - Vite)

*   **React.js:** A JavaScript library for building user interfaces.
*   **Vite:** A fast frontend build tool that provides a lightning-fast development experience.
*   **CSS:** For styling the application, ensuring a consistent and modern look.

### Machine Learning Models

*   `segmentation_model.h5`: A ResUnet++ model for segmenting lung regions from X-ray images.
*   `best_efficientnetb0_stage2_rgb_fixed.h5`: An EfficientNetB0 model trained for the classification of TB from segmented chest X-ray images.

## Setup and Installation

Follow these steps to get the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   Node.js (LTS version recommended)
*   npm or Yarn package manager

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/TB_WEB_APP.git
cd TB_WEB_APP
```

### 2. Backend Setup

Navigate to the `backend` directory, create a virtual environment, install dependencies, and start the FastAPI server.

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate # On Windows
# source venv/bin/activate # On macOS/Linux

pip install -r requirements.txt
uvicorn app:app --reload
```

The backend server will start on `http://127.0.0.1:8000`.

### 3. Frontend Setup

Open a new terminal, navigate to the `tb-cxr-frontend` directory, install dependencies, and start the React development server.

```bash
cd tb-cxr-frontend
npm install # or yarn install
npm run dev # or yarn dev
```

The frontend application will typically open in your browser at `http://localhost:5173` (or another port if 5173 is in use).

## Usage

1.  **Upload Image:** On the Dashboard page, drag and drop a chest X-ray image into the designated area or click "Browse Files" to select one.
2.  **Run Prediction:** Click the "Run Prediction" button.
3.  **View Results:** The application will display:
    *   The classification result (Normal/TB) and confidence score.
    *   The original X-ray image.
    *   The segmented lung regions.
    *   A Grad-CAM heatmap overlaying the segmented lung, indicating areas of importance for the classification.

## API Endpoints

### `POST /api/predict`

This endpoint accepts a chest X-ray image and returns the classification, segmented lung image, and Grad-CAM heatmap.

*   **Method:** `POST`
*   **Content-Type:** `multipart/form-data`
*   **Body:**
    *   `image`: The chest X-ray image file.
*   **Response (JSON):**
    ```json
    {
      "classification": {
        "label": "Normal",
        "confidence": 0.9876
      },
      "original_image": "base64_encoded_original_image_png",
      "segmented_lungs": "base64_encoded_segmented_lungs_png",
      "gradcam_on_segmented": "base64_encoded_gradcam_image_png"
    }
    ```

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please reach out to:

*   **Sheshank Singh**
*   **Email:** sheshanksingh2609@gmail.com
