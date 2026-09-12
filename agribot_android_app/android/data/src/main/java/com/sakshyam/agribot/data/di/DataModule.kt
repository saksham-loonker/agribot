package com.sakshyam.agribot.data.di

import android.content.Context
import androidx.room.Room
import com.sakshyam.agribot.data.db.AgribotDatabase
import com.sakshyam.agribot.data.repository.FileExportRepository
import com.sakshyam.agribot.data.repository.FileEvidenceRepository
import com.sakshyam.agribot.data.repository.RoomFieldLayoutRepository
import com.sakshyam.agribot.data.repository.RoomRunRepository
import com.sakshyam.agribot.data.settings.ScanSettingsDataStore
import com.sakshyam.agribot.domain.repository.EvidenceRepository
import com.sakshyam.agribot.domain.repository.ExportRepository
import com.sakshyam.agribot.domain.repository.FieldLayoutRepository
import com.sakshyam.agribot.domain.repository.RunRepository
import com.sakshyam.agribot.domain.repository.ScanSettingsRepository
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton
import javax.inject.Named
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {
    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): AgribotDatabase =
        Room.databaseBuilder(context, AgribotDatabase::class.java, "agribot.db")
            .addMigrations(AgribotDatabase.MIGRATION_1_2)
            .addMigrations(AgribotDatabase.MIGRATION_2_3)
            .addMigrations(AgribotDatabase.MIGRATION_3_4)
            .build()

    @Provides
    @Named("io")
    fun provideIoDispatcher(): CoroutineDispatcher = Dispatchers.IO
}

@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {
    @Binds
    abstract fun bindFieldLayoutRepository(repository: RoomFieldLayoutRepository): FieldLayoutRepository

    @Binds
    abstract fun bindRunRepository(repository: RoomRunRepository): RunRepository

    @Binds
    abstract fun bindExportRepository(repository: FileExportRepository): ExportRepository

    @Binds
    abstract fun bindEvidenceRepository(repository: FileEvidenceRepository): EvidenceRepository

    @Binds
    abstract fun bindScanSettingsRepository(repository: ScanSettingsDataStore): ScanSettingsRepository
}
