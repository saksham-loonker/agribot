package com.sakshyam.agribot.ml.di

import com.sakshyam.agribot.domain.scan.LeafVision
import com.sakshyam.agribot.ml.vision.AgribotLeafVision
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Qualifier
import javax.inject.Singleton
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import java.util.concurrent.Executors

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class InferenceDispatcher

@Module
@InstallIn(SingletonComponent::class)
abstract class MlModule {
    @Binds
    @Singleton
    abstract fun bindLeafVision(vision: AgribotLeafVision): LeafVision
}

@Module
@InstallIn(SingletonComponent::class)
object MlRuntimeModule {
    /** LiteRT interpreters are stateful: all model work runs on one background lane, never on Main. */
    @Provides
    @Singleton
    @InferenceDispatcher
    fun provideInferenceDispatcher(): CoroutineDispatcher = Executors.newSingleThreadExecutor { task ->
        Thread(task, "AgribotInference").apply { isDaemon = true }
    }.asCoroutineDispatcher()
}
