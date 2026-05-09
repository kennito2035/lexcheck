import Link from "next/link"
import styles from "./home.module.css"

export default function HomePage() {
  return (
    <>
      <section className={styles.hero}>
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>LexCheck</h1>
          <p className={styles.heroSubtitle}>
            Multimodal Dyslexia Screening Tool
          </p>
          <p className={styles.heroDescription}>
            Upload handwriting and eye-tracking data to generate a comprehensive screening report powered by AI and machine learning.
          </p>
          <Link href="/test" className={styles.ctaButton}>
            Start Screening
          </Link>
        </div>
      </section>

      <section className={styles.features}>
        <h2 className={styles.sectionTitle}>How It Works</h2>
        <div className={styles.featureGrid}>
          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>✏️</div>
            <h3>Handwriting Recognition</h3>
            <p>Analyzes letter formation, proportions, and writing patterns for indicators of dyslexia.</p>
          </div>
          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>👁️</div>
            <h3>Eye-Tracking Analysis</h3>
            <p>Tracks fixation patterns and saccade behavior to identify reading difficulties.</p>
          </div>
        </div>
      </section>

      <section className={styles.research}>
        <h2 className={styles.sectionTitle}>Research-Based</h2>
        <div className={styles.researchContent}>
          <p>
            LexCheck combines state-of-the-art computer vision (YOLOv11), machine learning (Random Forest), and explainable AI (SHAP, Grad-CAM) with large language model narration to provide clear, evidence-based screening recommendations.
          </p>
          <p className={styles.disclaimer}>
            <strong>Important Disclaimer:</strong> This tool provides risk indicators and is <strong>not a diagnosis</strong>. Results should be reviewed by qualified professionals. Always consult with a healthcare provider for official diagnosis.
          </p>
        </div>
      </section>

      <section className={styles.cta}>
        <h2>Ready to Get Started?</h2>
        <p>Upload your samples and receive an instant screening report.</p>
        <Link href="/test" className={styles.ctaButtonLarge}>
          Begin Screening
        </Link>
      </section>
    </>
  )
}
