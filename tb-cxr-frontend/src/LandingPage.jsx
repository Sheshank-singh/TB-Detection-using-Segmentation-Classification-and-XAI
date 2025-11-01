import React from 'react';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css'; // Import the new CSS file

function LandingPage() {
  const navigate = useNavigate();

  const handleTryDetectionClick = () => {
    navigate('/dashboard');
  };

  return (
    <div className="landing-page">
      {/* 1. Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <h1><span className="icon">🩺</span> AI-Powered Tuberculosis Detection from Chest X-rays</h1>
          <p>
            A deep learning-based diagnostic tool integrating lung segmentation, disease classification, and explainable AI (Grad-CAM) for reliable and transparent TB screening.
          </p>
          <div className="hero-buttons">
            <button className="button primary" onClick={handleTryDetectionClick}>Try Detection</button>
            <button className="button secondary">Learn More</button>
          </div>
        </div>
        <div className="hero-visual">
          <img src="/lung_image.png" alt="Workflow Transition" />
        </div>
      </section>

      {/* 2. About the Project */}
      <section className="about-section">
        <h2>About the Project</h2>
        <p>
          Tuberculosis (TB) remains one of the leading causes of death worldwide. Our system leverages deep learning to analyze chest X-rays, automatically identifying lung regions and detecting TB with high accuracy. The model combines:
        </p>
        <ul>
          <li><strong>ResUNet++</strong> for precise lung segmentation</li>
          <li><strong>EfficientNetB0</strong> for lightweight, high-accuracy classification</li>
          <li><strong>Grad-CAM</strong> for visual interpretability and medical transparency</li>
        </ul>
        <p>
          Achieved 98% segmentation accuracy (Dice: 0.99) and &gt;80% classification accuracy on benchmark datasets.
        </p>
      </section>

      {/* 3. System Workflow Diagram (Pipeline Section) */}
      <section className="workflow-section">
        <h2>System Workflow Diagram</h2>
        <div className="workflow-steps">
          <div className="workflow-step">
            <span className="icon">🖼️</span>
            <h3>Input Chest X-ray</h3>
            <p>Upload a chest X-ray image for analysis.</p>
          </div>
          <span className="arrow">→</span>
          <div className="workflow-step">
            <span className="icon">✂️</span>
            <h3>Lung Segmentation (ResUNet++)</h3>
            <p>Precisely isolate lung regions from the X-ray.</p>
          </div>
          <span className="arrow">→</span>
          <div className="workflow-step">
            <span className="icon">🔍</span>
            <h3>Post-Processing</h3>
            <p>Enhances mask quality using morphological filtering and hole filling.</p>
          </div>
          <span className="arrow">→</span>
          <div className="workflow-step">
            <span className="icon">🔬</span>
            <h3>Classification (EfficientNetB0)</h3>
            <p>Classify the cropped lung image for TB detection.</p>
          </div>
          <span className="arrow">→</span>
          <div className="workflow-step">
            <span className="icon">💡</span>
            <h3>Explainable AI (Grad-CAM)</h3>
            <p>Generate heatmaps to visualize model's decision regions.</p>
          </div>
        </div>
      </section>

      {/* 4. Key Results / Model Performance */}
      <section className="results-section">
        <h2>Key Results / Model Performance</h2>
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Segmentation (ResUNet++)</th>
              <th>Classification (EfficientNetB0)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Dice Coefficient</td>
              <td>0.96</td>
              <td>—</td>
            </tr>
            <tr>
              <td>Jaccard Index</td>
              <td>0.93</td>
              <td>—</td>
            </tr>
            <tr>
              <td>Accuracy</td>
              <td>0.98</td>
              <td>0.79</td>
            </tr>
            <tr>
              <td>Precision</td>
              <td>0.97</td>
              <td>0.80</td>
            </tr>
            <tr>
              <td>Recall</td>
              <td>0.96</td>
              <td>0.79</td>
            </tr>
          </tbody>
        </table>
        <p>The segmentation model ensures clean lung isolation before classification, enhancing diagnostic accuracy and interpretability.</p>
      </section>

      {/* 5. Explainable AI (Grad-CAM Section) */}
      <section className="xai-section">
        <h2>Explainable AI (Grad-CAM)</h2>
        <p>
          Explainable AI (XAI) provides visual insights into why the model makes a particular decision. Using Grad-CAM, our system highlights infected lung regions associated with TB — making it easier for clinicians to verify AI findings.
        </p>
        <div className="xai-examples">
          <div className="xai-example">
            <h3>Original</h3>
            <img src="tb_lung_img.png" alt="Original X-ray" />
          </div>
          <div className="xai-example">
            <h3>Segmented</h3>
            <img src="tb_seg_img.png" alt="Segmented Lung" />
          </div>
          <div className="xai-example">
            <h3>Grad-CAM Overlay</h3>
            <img src="tb_gradcam_img.png" alt="Grad-CAM Overlay" />
          </div>
        </div>
      </section>

      {/* 6. Datasets Used */}
      <section className="datasets-section">
        <h2>Datasets Used</h2>
        <ul>
          <li>Montgomery County X-ray Dataset (NIH, USA)</li>
          <li>Shenzhen Dataset (China’s CDC)</li>
        </ul>
      </section>

      {/* 7. Technologies & Architecture */}
      <section className="tech-section">
        <h2>Technologies & Architecture</h2>
        <table>
          <thead>
            <tr>
              <th>Component</th>
              <th>Tool/Library</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>Framework</td><td>TensorFlow / Keras</td></tr>
            <tr><td>Image Processing</td><td>OpenCV, NumPy</td></tr>
            <tr><td>Model</td><td>ResUNet++ + EfficientNetB0</td></tr>
            <tr><td>XAI</td><td>Grad-CAM</td></tr>
            <tr><td>Frontend</td><td>React + Tailwind CSS</td></tr>
            <tr><td>Backend</td><td>Flask / FastAPI</td></tr>
            <tr><td>Deployment</td><td>Hugging Face / Render</td></tr>
          </tbody>
        </table>
      </section>

      {/* 8. Impact & Applications */}
      <section className="impact-section">
        <h2>Impact & Applications</h2>
        <p>This solution can support:</p>
        <ul>
          <li>Early TB screening in rural/remote regions</li>
          <li>Assisting radiologists in faster diagnosis</li>
          <li>Reducing human error and improving workflow efficiency</li>
        </ul>
      </section>

      {/* 9. About Research / Credits */}
      <section className="research-section">
        <h2>About Research / Credits</h2>
        <p>
          This project is part of ongoing research in AI-assisted medical imaging under the mentorship of [Professor’s Name] at Vidyalankar Institute of Technology. Current work includes improving interpretability and extending the model to multi-disease detection.
        </p>
        <div className="research-links">
          <a href="#" className="button secondary">Read Research Summary (PDF)</a>
          <a href="#" className="button secondary">View GitHub Repository</a>
        </div>
      </section>

      {/* 10. Footer */}
      <footer className="footer-section">
        <p>Contact: sheshanksingh2609@gmail.com</p>
        <div className="social-links">
          <a href="https://github.com/Sheshank-singh/" target="_blank" rel="noopener noreferrer" className="social-link">
            <img src="github.png" alt="GitHub" className="social-icon" />
            GitHub
          </a>
          <a href="https://www.linkedin.com/in/sheshanksingh2609" target="_blank" rel="noopener noreferrer" className="social-link">
            <img src="linkedin.png" alt="LinkedIn" className="social-icon" />
            LinkedIn
          </a>
          <a href="https://www.kaggle.com/code/sheshanksingh" target="_blank" rel="noopener noreferrer" className="social-link">
            <img src="kaggle.png" alt="Kaggle" className="social-icon" />
            Kaggle
          </a>
          <a href="http://huggingface/sheshank-singh" target="_blank" rel="noopener noreferrer" className="social-link">
            <img src="huggingface.png" alt="Hugging Face" className="social-icon" />
            Hugging Face
          </a>
        </div>
        <p className="disclaimer">
          Disclaimer: This tool is intended for educational and research purposes. It should not replace professional medical diagnosis.
        </p>
      </footer>
    </div>
  );
}

export default LandingPage;
