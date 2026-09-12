package com.sakshyam.agribot.data.db

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [
        FieldLayoutEntity::class,
        FieldRowEntity::class,
        ScanRunEntity::class,
        DecisionEntity::class,
        FrontGeometryEntity::class,
        RunEventEntity::class,
        ModelBundleEntity::class,
    ],
    version = 4,
    exportSchema = true,
)
abstract class AgribotDatabase : RoomDatabase() {
    abstract fun fieldLayoutDao(): FieldLayoutDao
    abstract fun runDao(): RunDao
    abstract fun decisionDao(): DecisionDao
    abstract fun runEventDao(): RunEventDao
    abstract fun modelBundleDao(): ModelBundleDao

    companion object {
        val MIGRATION_1_2: Migration = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE decisions ADD COLUMN measurementDistanceM REAL")
                db.execSQL("ALTER TABLE decisions ADD COLUMN measurementSource TEXT")
                db.execSQL("ALTER TABLE decisions ADD COLUMN measurementQuality TEXT")
                db.execSQL("ALTER TABLE decisions ADD COLUMN relativePlantWidth REAL")
                db.execSQL("ALTER TABLE decisions ADD COLUMN relativePlantHeight REAL")
                db.execSQL("ALTER TABLE decisions ADD COLUMN sizeSource TEXT")
                db.execSQL("ALTER TABLE decisions ADD COLUMN sizeQuality TEXT")
                db.execSQL("ALTER TABLE decisions ADD COLUMN gpsAccuracyM REAL")
                db.execSQL("ALTER TABLE decisions ADD COLUMN gpsFixAgeSeconds REAL")
                db.execSQL("ALTER TABLE decisions ADD COLUMN top2Margin REAL")
                db.execSQL("ALTER TABLE decisions ADD COLUMN predictionEntropy REAL")
            }
        }

        val MIGRATION_2_3: Migration = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE decisions ADD COLUMN trackId TEXT")
            }
        }

        val MIGRATION_3_4: Migration = object : Migration(3, 4) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE decisions ADD COLUMN treatmentStatus TEXT NOT NULL DEFAULT 'NOT_TREATED'")
                db.execSQL("ALTER TABLE decisions ADD COLUMN treatmentNote TEXT")
            }
        }
    }
}
