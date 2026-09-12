# Agribot UI and Plant Detection Fix Plan

## Current Issues Analysis

### 1. Sideways Scan Problems
- **No Individual Plant Boxing**: The current system doesn't create distinct bounding boxes for each plant
- **Poor Plant Tracking**: Plants aren't properly identified and tracked across frames
- **Missing Visual Feedback**: No visual indication of detected plants during scanning

### 2. UI/UX Problems
- **No Screen Separation**: Recording and data viewing share the same interface
- **Confusing Workflow**: Users can't easily switch between scanning and analyzing data
- **Poor Visual Design**: Interface lacks modern UI principles and intuitive controls

### 3. Plant Detection Issues
- **No Unique Plant IDs**: Plants aren't assigned distinct identifiers for tracking
- **Basic Detection Only**: Current system uses simple frame-by-frame analysis without tracking
- **No Sensor Integration**: Accelerometer and tilt data aren't used for better positioning

### 4. Technical Debt
- **Hardcoded Values**: Many configuration values are hardcoded
- **Tight Coupling**: UI and business logic are tightly coupled
- **Limited Error Handling**: Poor error recovery mechanisms

## Comprehensive Solution

### Phase 1: Enhanced Plant Detection System

#### 1.1 Implement Cosine Similarity for Plant Matching
```kotlin
// Add to TFLiteDetector.kt or new PlantTracker.kt
fun calculateCosineSimilarity(feat1: FloatArray, feat2: FloatArray): Float {
    var dotProduct = 0f
    var norm1 = 0f
    var norm2 = 0f
    
    for (i in feat1.indices) {
        dotProduct += feat1[i] * feat2[i]
        norm1 += feat1[i] * feat1[i]
        norm2 += feat2[i] * feat2[i]
    }
    
    return dotProduct / (sqrt(norm1) * sqrt(norm2))
}
```

#### 1.2 Add IoU (Intersection over Union) Tracking
```kotlin
data class TrackedPlant(
    val plantId: String,
    val currentBbox: BoundingBox,
    val history: List<BoundingBox>,
    val features: FloatArray,
    val lastSeenFrame: Long
)

fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
    val intersectionWidth = max(0f, min(box1.right, box2.right) - max(box1.left, box2.left))
    val intersectionHeight = max(0f, min(box1.bottom, box2.bottom) - max(box1.top, box2.top))
    val intersectionArea = intersectionWidth * intersectionHeight
    
    val box1Area = box1.width * box1.height
    val box2Area = box2.width * box2.height
    
    return intersectionArea / (box1Area + box2Area - intersectionArea)
}
```

#### 1.3 Unique Plant ID System
```kotlin
class PlantIdGenerator {
    private var nextId = 1
    
    fun generateId(): String {
        return "plant_${nextId++}"
    }
    
    fun generateIdWithPosition(x: Float, y: Float, timestamp: Long): String {
        // Create spatially and temporally unique ID
        val xHash = (x * 1000).toInt() % 1000
        val yHash = (y * 1000).toInt() % 1000
        val timeHash = (timestamp % 10000)
        return "p_${xHash}_${yHash}_${timeHash}_${nextId++}"
    }
}
```

### Phase 2: Sensor Integration

#### 2.1 Accelerometer and Tilt Integration
```kotlin
// Add to SideScanViewModel.kt
class SensorFusionManager(context: Context) : SensorEventListener {
    private val sensorManager = context.getSystemService(SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    
    private val gravity = FloatArray(3)
    private val geomagnetic = FloatArray(3)
    private val rotationMatrix = FloatArray(9)
    private val orientation = FloatArray(3)
    
    var currentTilt: Float = 0f
    var currentRoll: Float = 0f
    var currentPitch: Float = 0f
    
    fun start() {
        sensorManager.registerListener(
            this, 
            accelerometer, 
            SensorManager.SENSOR_DELAY_NORMAL
        )
        sensorManager.registerListener(
            this, 
            gyroscope, 
            SensorManager.SENSOR_DELAY_NORMAL
        )
    }
    
    fun stop() {
        sensorManager.unregisterListener(this)
    }
    
    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                // Update gravity and calculate orientation
                SensorManager.getRotationMatrix(rotationMatrix, null, 
                    event.values, geomagnetic)
                SensorManager.getOrientation(rotationMatrix, orientation)
                
                currentRoll = orientation[2]  // Z-axis rotation
                currentPitch = orientation[1] // X-axis rotation
                currentTilt = sqrt(orientation[1] * orientation[1] + 
                                 orientation[2] * orientation[2])
            }
        }
    }
    
    fun getTiltCompensation(): FloatArray {
        // Calculate compensation factors for plant positioning
        return floatArrayOf(
            cos(currentPitch), 
            sin(currentPitch), 
            cos(currentRoll)
        )
    }
}
```

#### 2.2 3D Position Mapping
```kotlin
data class PlantPosition3D(
    val plantId: String,
    val x2d: Float,       // 2D screen position
    val y2d: Float,       // 2D screen position
    val x3d: Float,       // 3D world position (tilt-compensated)
    val y3d: Float,       // 3D world position (tilt-compensated)
    val z3d: Float,       // 3D world position (distance estimate)
    val timestamp: Long,
    val confidence: Float
)

fun map2DTo3D(
    bbox: BoundingBox,
    frameWidth: Int,
    frameHeight: Int,
    tilt: Float,
    cameraHeight: Float = 1.5f
): PlantPosition3D {
    // Convert screen coordinates to 3D world coordinates
    val centerX = bbox.centerX / frameWidth
    val centerY = bbox.centerY / frameHeight
    
    // Apply tilt compensation
    val x3d = (centerX - 0.5f) * cos(tilt)
    val y3d = (0.5f - centerY) * sin(tilt)
    val z3d = cameraHeight - (0.5f - centerY) * cos(tilt)
    
    return PlantPosition3D(
        plantId = "temp",
        x2d = bbox.centerX,
        y2d = bbox.centerY,
        x3d = x3d,
        y3d = y3d,
        z3d = z3d,
        timestamp = System.currentTimeMillis(),
        confidence = bbox.width * bbox.height / (frameWidth * frameHeight)
    )
}
```

### Phase 3: UI/UX Redesign

#### 3.1 Recording Screen with Live Bounding Boxes
```kotlin
@Composable
fun EnhancedRecordingScreen(
    state: ScanUiState,
    viewModel: SideScanViewModel,
    trackedPlants: List<TrackedPlant>
) {
    Box(modifier = Modifier.fillMaxSize()) {
        // Camera preview
        CameraPreview(
            modifier = Modifier.fillMaxSize(),
            targetFps = state.targetFps,
            onFrame = viewModel::onAnalysisFrame
        )
        
        // Bounding box overlay
        Canvas(modifier = Modifier.fillMaxSize()) {
            trackedPlants.forEach { plant ->
                drawRect(
                    color = when (plant.healthStatus) {
                        "Healthy" -> Color.Green.copy(alpha = 0.6f)
                        "Sick" -> Color.Red.copy(alpha = 0.6f)
                        else -> Color.Yellow.copy(alpha = 0.6f)
                    },
                    topLeft = Offset(plant.currentBbox.left, plant.currentBbox.top),
                    size = Size(plant.currentBbox.width, plant.currentBbox.height),
                    style = Stroke(width = 4f)
                )
                
                // Draw plant ID
                drawText(
                    text = plant.plantId,
                    color = Color.White,
                    topLeft = Offset(plant.currentBbox.left + 4f, plant.currentBbox.top + 20f),
                    fontSize = 16f
                )
                
                // Draw confidence
                drawText(
                    text = "${(plant.confidence * 100).toInt()}%",
                    color = Color.White,
                    topLeft = Offset(plant.currentBbox.left + 4f, plant.currentBbox.top + 40f),
                    fontSize = 14f
                )
            }
        }
        
        // Sensor data overlay
        Column(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(16.dp)
                .background(Color.Black.copy(alpha = 0.7f))
                .padding(8.dp)
        ) {
            Text("Tilt: ${String.format("%.1f", state.sensorData.tilt)}°", color = Color.White)
            Text("Roll: ${String.format("%.1f", state.sensorData.roll)}°", color = Color.White)
            Text("Pitch: ${String.format("%.1f", state.sensorData.pitch)}°", color = Color.White)
            Text("Plants: ${trackedPlants.size}", color = Color.White)
        }
        
        // Control buttons
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Button(
                onClick = viewModel::toggleRecording,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (state.isRecording) "Stop Recording" else "Start Recording")
            }
            
            Row(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Button(
                    onClick = { /* Manual override */ },
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Mark Healthy")
                }
                Button(
                    onClick = { /* Manual override */ },
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Mark Sick")
                }
            }
        }
    }
}
```

#### 3.2 Data Analysis Screen
```kotlin
@Composable
fun DataAnalysisScreen(
    viewModel: SideScanViewModel,
    onNavigateToRecording: () -> Unit
) {
    val plants by viewModel.trackedPlants.collectAsState()
    val stats by viewModel.scanStatistics.collectAsState()
    
    Column(modifier = Modifier.fillMaxSize()) {
        // Header with navigation
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Scan Results", style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.weight(1f))
            Button(onClick = onNavigateToRecording) {
                Text("Back to Scan")
            }
        }
        
        // Statistics summary
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Scan Summary", style = MaterialTheme.typography.titleLarge)
                Spacer(Modifier.height(8.dp))
                Row {
                    Text("Total Plants:")
                    Spacer(Modifier.weight(1f))
                    Text(stats.totalPlants.toString(), fontWeight = FontWeight.Bold)
                }
                Row {
                    Text("Healthy:")
                    Spacer(Modifier.weight(1f))
                    Text("${stats.healthyCount} (${stats.healthyPercent}%)", 
                        color = Color.Green, fontWeight = FontWeight.Bold)
                }
                Row {
                    Text("Sick:")
                    Spacer(Modifier.weight(1f))
                    Text("${stats.sickCount} (${stats.sickPercent}%)", 
                        color = Color.Red, fontWeight = FontWeight.Bold)
                }
                Row {
                    Text("Uncertain:")
                    Spacer(Modifier.weight(1f))
                    Text("${stats.uncertainCount} (${stats.uncertainPercent}%)", 
                        color = Color.Yellow, fontWeight = FontWeight.Bold)
                }
            }
        }
        
        // Filter controls
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            FilterChip(
                selected = viewModel.filterType == "all",
                onClick = { viewModel.setFilter("all") },
                label = { Text("All Plants") }
            )
            FilterChip(
                selected = viewModel.filterType == "healthy",
                onClick = { viewModel.setFilter("healthy") },
                label = { Text("Healthy") }
            )
            FilterChip(
                selected = viewModel.filterType == "sick",
                onClick = { viewModel.setFilter("sick") },
                label = { Text("Sick") }
            )
            FilterChip(
                selected = viewModel.filterType == "uncertain",
                onClick = { viewModel.setFilter("uncertain") },
                label = { Text("Uncertain") }
            )
        }
        
        // Plant grid view
        LazyVerticalGrid(
            columns = GridCells.Adaptive(minSize = 120.dp),
            modifier = Modifier.weight(1f)
        ) {
            items(plants.filter { viewModel.filterMatches(it) }) { plant ->
                PlantCard(plant = plant, onClick = { viewModel.selectPlant(plant.plantId) })
            }
        }
    }
}

@Composable
fun PlantCard(plant: TrackedPlant, onClick: () -> Unit) {
    Card(
        onClick = onClick,
        modifier = Modifier.padding(8.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Plant thumbnail (from evidence frames)
            if (plant.thumbnail != null) {
                Image(
                    bitmap = plant.thumbnail.asImageBitmap(),
                    contentDescription = "Plant ${plant.plantId}",
                    modifier = Modifier
                        .size(80.dp)
                        .clip(RoundedCornerShape(8.dp))
                )
            } else {
                Box(
                    modifier = Modifier
                        .size(80.dp)
                        .background(Color.Gray)
                        .clip(RoundedCornerShape(8.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    Text("No Image", color = Color.White)
                }
            }
            
            Spacer(Modifier.height(8.dp))
            
            Text(plant.plantId, fontWeight = FontWeight.Bold)
            
            // Health status indicator
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(top = 4.dp)
            ) {
                Icon(
                    imageVector = when (plant.healthStatus) {
                        "Healthy" -> Icons.Default.CheckCircle
                        "Sick" -> Icons.Default.Warning
                        else -> Icons.Default.Help
                    },
                    contentDescription = null,
                    tint = when (plant.healthStatus) {
                        "Healthy" -> Color.Green
                        "Sick" -> Color.Red
                        else -> Color.Yellow
                    }
                )
                Spacer(Modifier.width(4.dp))
                Text(plant.healthStatus)
            }
            
            Text("${(plant.confidence * 100).toInt()}% conf", 
                style = MaterialTheme.typography.labelSmall)
        }
    }
}
```

### Phase 4: Database and Data Management

#### 4.1 Plant Tracking Database
```kotlin
// PlantTrackingDatabase.kt
@Entity(tableName = "tracked_plants")
data class TrackedPlantEntity(
    @PrimaryKey val plantId: String,
    val sessionId: String,
    val firstSeenTimestamp: Long,
    val lastSeenTimestamp: Long,
    val healthStatus: String,
    val confidence: Float,
    val totalFrames: Int,
    val averagePositionX: Float,
    val averagePositionY: Float,
    val minBoundingBox: String, // JSON serialized
    val maxBoundingBox: String, // JSON serialized
    val featuresVector: String, // Base64 encoded
    val manualOverride: Boolean
)

@Entity(tableName = "plant_positions")
data class PlantPositionEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val plantId: String,
    val sessionId: String,
    val timestamp: Long,
    val frameIndex: Int,
    val positionX: Float,
    val positionY: Float,
    val positionZ: Float,
    val boundingBox: String, // JSON serialized
    val confidence: Float,
    val sensorTilt: Float,
    val sensorRoll: Float,
    val sensorPitch: Float
)

@Dao
interface PlantTrackingDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPlant(plant: TrackedPlantEntity)
    
    @Insert
    suspend fun insertPosition(position: PlantPositionEntity)
    
    @Query("SELECT * FROM tracked_plants WHERE sessionId = :sessionId")
    fun getPlantsBySession(sessionId: String): Flow<List<TrackedPlantEntity>>
    
    @Query("SELECT * FROM plant_positions WHERE plantId = :plantId ORDER BY timestamp")
    fun getPositionsForPlant(plantId: String): Flow<List<PlantPositionEntity>>
    
    @Query("DELETE FROM tracked_plants WHERE sessionId = :sessionId")
    suspend fun deletePlantsBySession(sessionId: String)
    
    @Query("DELETE FROM plant_positions WHERE sessionId = :sessionId")
    suspend fun deletePositionsBySession(sessionId: String)
}

// PlantTrackingRepository.kt
class PlantTrackingRepository(private val dao: PlantTrackingDao) {
    fun trackPlant(plant: TrackedPlant, sessionId: String) {
        viewModelScope.launch {
            // Convert to entity and save
            val entity = plant.toEntity(sessionId)
            dao.insertPlant(entity)
            
            // Save current position
            val position = plant.currentPosition.toPositionEntity(sessionId)
            dao.insertPosition(position)
        }
    }
    
    fun getAllPlants(sessionId: String): Flow<List<TrackedPlant>> {
        return dao.getPlantsBySession(sessionId).map { entities ->
            entities.map { it.toDomainModel() }
        }
    }
    
    fun getPlantHistory(plantId: String): Flow<List<PlantPosition>> {
        return dao.getPositionsForPlant(plantId).map { entities ->
            entities.map { it.toDomainModel() }
        }
    }
}
```

### Phase 5: Implementation Roadmap

#### Step 1: Setup Plant Tracking System (2-3 days)
- Create `PlantTracker.kt` with cosine similarity and IoU tracking
- Implement unique plant ID generation
- Add sensor fusion manager
- Create 3D position mapping

#### Step 2: Enhance ML Detection (3-4 days)
- Modify `TFLiteDetector.kt` to return bounding boxes
- Add non-max suppression for overlapping detections
- Implement feature extraction for plant matching
- Add confidence scoring system

#### Step 3: Database Implementation (2 days)
- Create Room database for plant tracking
- Implement DAO and repository patterns
- Add data migration system
- Implement export/import functionality

#### Step 4: UI Redesign (4-5 days)
- Create new recording screen with bounding box overlay
- Design data analysis screen with filters and grid
- Implement navigation system
- Add sensor data visualization

#### Step 5: Integration and Testing (3-4 days)
- Connect plant tracking with UI
- Test complete workflow
- Optimize performance
- Fix bugs and edge cases

#### Step 6: Field Testing (2-3 days)
- Test with real plants
- Validate detection accuracy
- Test various lighting conditions
- Gather user feedback

### Expected Outcomes

1. **Improved Plant Detection**: Each plant will have distinct bounding box with unique ID
2. **Better Tracking**: Plants will be consistently tracked across frames using multiple algorithms
3. **Enhanced UI**: Clear separation between recording and analysis modes
4. **Sensor Integration**: Accelerometer and tilt data will improve positioning accuracy
5. **Data Management**: Complete history of all plants with export capabilities

### Success Metrics

- ✅ **Detection Accuracy**: >90% of plants correctly identified and tracked
- ✅ **UI Satisfaction**: Users can easily navigate between recording and analysis
- ✅ **Performance**: Real-time processing at 5+ FPS on mid-range devices
- ✅ **Reliability**: <5% tracking errors in field conditions
- ✅ **Usability**: Farmers can use the app with minimal training

This comprehensive plan addresses all the fundamental issues with the current implementation and provides a clear path to a much more robust and user-friendly system.