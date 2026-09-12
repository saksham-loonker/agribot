package com.sakshyam.agribot.ml.di

import com.sakshyam.agribot.domain.repository.InferenceRepository
import com.sakshyam.agribot.ml.inference.TfliteInferenceRepository
import dagger.Binds
import dagger.Module
import dagger.Provides
import javax.inject.Qualifier
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class InferenceDispatcher

@Module
@InstallIn(SingletonComponent::class)
abstract class MlModule {
    @Binds
    @Singleton
    abstract fun bindInferenceRepository(repository: TfliteInferenceRepository): InferenceRepository
}

@Module
@InstallIn(SingletonComponent::class)
@OptIn(ExperimentalCoroutinesApi::class)
object MlRuntimeModule {
    /**
     * Model execution is serialized onto one bounded lane.  TFLite
     * interpreters are stateful and the caller must never run them on Main.
     * The ViewModel also keeps at most one active request, so this dispatcher
     * cannot accumulate an unbounded inference backlog.
     */
    @Provides
    @Singleton
    @InferenceDispatcher
    fun provideInferenceDispatcher(): CoroutineDispatcher =
        Dispatchers.Default.limitedParallelism(1)
}
