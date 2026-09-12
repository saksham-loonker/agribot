# Native Agribot release rules. Keep framework/runtime entry points that are
# discovered by generated code, JNI, or Android framework metadata.

-keepattributes RuntimeVisibleAnnotations,RuntimeVisibleParameterAnnotations,Signature,InnerClasses,EnclosingMethod

# TensorFlow Lite interpreter/runtime classes are called through JNI.
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# Hilt entry points and generated components.
-keep @dagger.hilt.android.HiltAndroidApp class * { *; }
-keep @dagger.hilt.android.lifecycle.HiltViewModel class * { *; }
-keep class dagger.hilt.** { *; }
-keep class hilt_aggregated_deps.** { *; }
-dontwarn dagger.hilt.**

# Room database/DAO metadata used by generated implementations.
-keep class * extends androidx.room.RoomDatabase { *; }
-keep @androidx.room.Database class * { *; }
-keep @androidx.room.Dao class * { *; }
-keep @androidx.room.Entity class * { *; }
-dontwarn androidx.room.**

# CameraX binds use cases through AndroidX runtime classes.
-keep class androidx.camera.** { *; }
-dontwarn androidx.camera.**

# Keep local app entry points explicit for release diagnostics.
-keep class com.sakshyam.agribot.AgribotApplication { *; }
-keep class com.sakshyam.agribot.MainActivity { *; }
