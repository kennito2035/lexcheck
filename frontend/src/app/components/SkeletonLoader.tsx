import styles from "./SkeletonLoader.module.css"

interface SkeletonBarProps {
  width?: string
  height?: string
}

interface SkeletonTextProps {
  lines?: number
  width?: string
}

interface SkeletonBoxProps {
  width?: string
  height?: string
}

export function SkeletonBar({ width = "100%", height = "24px" }: SkeletonBarProps) {
  return <div className={styles.skeleton} style={{ width, height }} />
}

export function SkeletonText({ lines = 3, width = "100%" }: SkeletonTextProps) {
  return (
    <div className={styles.skeletonTextContainer}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className={styles.skeleton}
          style={{
            width: i === lines - 1 ? "80%" : width,
            height: "16px",
            marginBottom: i < lines - 1 ? "8px" : "0"
          }}
        />
      ))}
    </div>
  )
}

export function SkeletonBox({ width = "100%", height = "200px" }: SkeletonBoxProps) {
  return <div className={styles.skeleton} style={{ width, height }} />
}
