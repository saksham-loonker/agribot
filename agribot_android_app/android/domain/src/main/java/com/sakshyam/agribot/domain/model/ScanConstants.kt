package com.sakshyam.agribot.domain.model

/**
 * Centralized constants for the Agribot scanning system.
 * All magic numbers and configuration values should be defined here.
 */
object ScanConstants {
    // Default performance settings
    const val DEFAULT_TARGET_FPS = 5
    const val DEFAULT_CPU_THREADS = 4
    const val DEFAULT_CONFIDENCE_THRESHOLD = 0.62f
    // Disease scores are temperature-scaled before this gate.  A raw softmax
    // score is not sufficient evidence for an actionable disease label.
    // The default is intentionally high: a misclassified "Late Blight" on a
    // healthy plant triggers a real-world treatment recommendation.
    const val DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.90f
    const val DEFAULT_CONFIDENCE_TEMPERATURE = 3.0f
    
    // Plant timing
    const val DEFAULT_PLANT_COOLDOWN_SEC = 2.0
    const val MIN_PLANT_COOLDOWN_SEC = 0.5
    const val MAX_PLANT_COOLDOWN_SEC = 10.0
    
    // Front Overview calibration defaults
    const val DEFAULT_CAMERA_HEIGHT_M = 1.1
    const val DEFAULT_CAMERA_DISTANCE_M = 1.0
    const val DEFAULT_CAMERA_TILT_DEGREES = 15.0
    const val DEFAULT_GUIDE_RAIL_LEFT_PX = 192f
    const val DEFAULT_GUIDE_RAIL_RIGHT_PX = 448f
    const val DEFAULT_FRAME_WIDTH_PX = 640
    const val DEFAULT_FRAME_HEIGHT_PX = 480
    const val DEFAULT_CALIBRATION_QUALITY = 0.70f
    
    // Model configuration
    const val DEFAULT_MODEL_BUNDLE_ID = "agribot-model-bundle-v001"
    // Detector score is intentionally high: YOLO fires on enough texture in
    // a lab to "see" non-plant objects at ~0.3 confidence, and the cost of
    // admitting one false plant is high (it becomes a recorded disease
    // decision). A higher bar trades a little recall for far fewer phantom
    // plants on indoor scenes.
    const val MIN_DETECTOR_CONFIDENCE = 0.50f
    // A bad frame must not turn into hundreds of classifier calls. Keep the
    // highest-scoring candidates so the UI remains responsive on a phone.
    const val MAX_DETECTOR_CANDIDATES = 16

    // Scene gates
    // Frames with very little vegetation coverage (lab, indoor, bare soil
    // before canopy) should not turn any detector box into a plant decision.
    const val MIN_VEGETATION_COVERAGE = 0.012f
    // If the detector returns boxes on a scene with too little vegetation,
    // we require this many high-confidence boxes to even consider it.
    const val MIN_PLANT_CANDIDATES_WHEN_BARE = 2
    
    // Frame processing
    const val MAX_FRAME_QUEUE_SIZE = 30
    const val MAX_EDGE_SIZE = 256
    const val DEFAULT_CROP_SCALE = 0.55f
    const val DEFAULT_CROP_PAD = 0.05f
    
    // Storage limits
    const val MAX_EVIDENCE_FRAMES = 100
    const val MAX_STORAGE_BYTES = 50L * 1024 * 1024  // 50 MB
}
