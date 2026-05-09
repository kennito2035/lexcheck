import styles from "./about.module.css"

export default function AboutPage() {
  return (
    <>
      <section className={styles.header}>
        <h1 className={styles.title}>About LexCheck</h1>
        <p className={styles.subtitle}>
          A Research-Based Multimodal Dyslexia Screening Tool
        </p>
      </section>

      <section className={styles.mission}>
        <h2>Our Mission</h2>
        <p>
          LexCheck empowers educators and clinicians with a fast, evidence-based screening tool that combines computer vision, machine learning, and natural language processing to identify potential dyslexia indicators through multimodal analysis.
        </p>
      </section>

      <section className={styles.features}>
        <h2>Key Features</h2>
        <div className={styles.featuresList}>
          <div className={styles.featureItem}>
            <h3>✏️ Handwriting Recognition</h3>
            <p>
              Advanced computer vision analyzes handwriting for indicators such as proportionality issues, letter formation problems, and spatial organization patterns.
            </p>
          </div>
          <div className={styles.featureItem}>
            <h3>👁️ Eye-Tracking Analysis</h3>
            <p>
              Random Forest classifier processes gaze data to identify reading difficulties through fixation patterns, saccade velocity, and regression rates. Achieves 88.58% accuracy on published datasets.
            </p>
          </div>
          <div className={styles.featureItem}>
            <h3>🔍 Explainable AI</h3>
            <p>
              SHAP (SHapley Additive exPlanations) values and Grad-CAM heatmaps provide transparent, interpretable results that show exactly which features contributed to the assessment.
            </p>
          </div>
          <div className={styles.featureItem}>
            <h3>🤖 AI Narration</h3>
            <p>
              Large language models generate plain-language summaries that translate complex model outputs into actionable insights for educators and clinicians.
            </p>
          </div>
          <div className={styles.featureItem}>
            <h3>📊 Comprehensive Report</h3>
            <p>
              Receive a detailed risk score (0-1 scale), key indicators, visualizations, and evidence-based narration in a single, easy-to-understand report.
            </p>
          </div>
        </div>
      </section>

      <section className={styles.research}>
        <h2>Research Foundation</h2>
        <p>
          LexCheck is grounded in peer-reviewed research and uses models trained on validated dyslexia screening data:
        </p>
        <ul className={styles.researchList}>
          <li>
            <strong>YOLOv11 for Handwriting:</strong> State-of-the-art object detection fine-tuned to identify reversal and spacing patterns (arXiv:2501.15263)
          </li>
          <li>
            <strong>Random Forest for Eye-Tracking:</strong> 12-feature classifier achieving 88.58% accuracy on reading difficulty prediction (Cogan & Ngo, 2025, arXiv:2506.11004)
          </li>
          <li>
            <strong>SHAP Explainability:</strong> TreeExplainer provides feature importance rankings to understand model decisions
          </li>
          <li>
            <strong>Grad-CAM Visualizations:</strong> Highlight which regions of handwriting triggered dyslexia indicators
          </li>
        </ul>
      </section>

      <section className={styles.howItWorks}>
        <h2>How It Works</h2>
        <div className={styles.processSteps}>
          <div className={styles.step}>
            <div className={styles.stepNumber}>1</div>
            <h3>Upload</h3>
            <p>Submit handwriting images and eye-tracking CSV data</p>
          </div>
          <div className={styles.stepArrow}>→</div>
          <div className={styles.step}>
            <div className={styles.stepNumber}>2</div>
            <h3>Analyze</h3>
            <p>AI models process data in parallel to detect indicators</p>
          </div>
          <div className={styles.stepArrow}>→</div>
          <div className={styles.step}>
            <div className={styles.stepNumber}>3</div>
            <h3>Report</h3>
            <p>Receive risk score, visualizations, and AI-generated summary</p>
          </div>
        </div>
      </section>

      <section className={styles.disclaimer}>
        <h2>Important Disclaimer</h2>
        <div className={styles.disclaimerBox}>
          <p>
            <strong>LexCheck is a screening tool, not a diagnosis.</strong> Results indicate potential dyslexia indicators and should be interpreted by qualified professionals. This tool:
          </p>
          <ul>
            <li>Does NOT provide medical diagnosis or treatment recommendations</li>
            <li>Should NOT replace professional evaluation by speech-language pathologists or dyslexia specialists</li>
            <li>Is designed to support initial screening and referral processes</li>
            <li>Provides probabilistic risk indicators based on machine learning models</li>
          </ul>
          <p>
            Always consult with qualified healthcare providers, educators, or dyslexia specialists for definitive diagnosis and treatment planning.
          </p>
        </div>
      </section>

      <section className={styles.contact}>
        <h2>Questions or Feedback?</h2>
        <p>
          LexCheck is a research tool created to support dyslexia screening. We welcome feedback and collaboration inquiries.
        </p>
      </section>
    </>
  )
}

