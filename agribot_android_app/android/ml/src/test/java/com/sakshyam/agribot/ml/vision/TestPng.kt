package com.sakshyam.agribot.ml.vision

import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.File
import java.util.zip.Inflater
import kotlin.math.abs

/** Minimal lossless PNG reader for tests (8-bit RGB/RGBA, non-interlaced) -> packed ARGB. */
internal object TestPng {
    fun read(file: File): RgbFrame {
        val inp = DataInputStream(file.inputStream().buffered())
        val sig = ByteArray(8).also { inp.readFully(it) }
        require(sig.contentEquals(byteArrayOf(-119, 80, 78, 71, 13, 10, 26, 10))) { "not a PNG" }
        var w = 0; var h = 0; var colorType = 0
        val idat = ByteArrayOutputStream()
        while (true) {
            val len = inp.readInt()
            val type = String(ByteArray(4).also { inp.readFully(it) }, Charsets.US_ASCII)
            val data = ByteArray(len).also { inp.readFully(it) }
            inp.readInt() // crc
            when (type) {
                "IHDR" -> {
                    w = ((data[0].toInt() and 0xFF) shl 24) or ((data[1].toInt() and 0xFF) shl 16) or ((data[2].toInt() and 0xFF) shl 8) or (data[3].toInt() and 0xFF)
                    h = ((data[4].toInt() and 0xFF) shl 24) or ((data[5].toInt() and 0xFF) shl 16) or ((data[6].toInt() and 0xFF) shl 8) or (data[7].toInt() and 0xFF)
                    require(data[8].toInt() == 8) { "bit depth must be 8" }
                    colorType = data[9].toInt()
                    require(colorType == 2 || colorType == 6) { "RGB/RGBA only" }
                    require(data[12].toInt() == 0) { "interlaced PNG not supported" }
                }
                "IDAT" -> idat.write(data)
                "IEND" -> break
            }
        }
        val bpp = if (colorType == 6) 4 else 3
        val stride = w * bpp
        val raw = ByteArray((stride + 1) * h)
        Inflater().apply { setInput(idat.toByteArray()); var off = 0; while (off < raw.size) off += inflate(raw, off, raw.size - off).also { require(it > 0 || !needsInput()) }; end() }
        val cur = ByteArray(stride); val prev = ByteArray(stride)
        val px = IntArray(w * h)
        for (y in 0 until h) {
            val f = raw[y * (stride + 1)].toInt()
            System.arraycopy(raw, y * (stride + 1) + 1, cur, 0, stride)
            for (i in 0 until stride) {
                val a = if (i >= bpp) cur[i - bpp].toInt() and 0xFF else 0
                val b = prev[i].toInt() and 0xFF
                val c = if (i >= bpp) prev[i - bpp].toInt() and 0xFF else 0
                val x = cur[i].toInt() and 0xFF
                val v = when (f) {
                    0 -> x
                    1 -> x + a
                    2 -> x + b
                    3 -> x + (a + b) / 2
                    4 -> { val p = a + b - c; val pa = abs(p - a); val pb = abs(p - b); val pc = abs(p - c); x + if (pa <= pb && pa <= pc) a else if (pb <= pc) b else c }
                    else -> error("bad filter $f")
                }
                cur[i] = v.toByte()
            }
            for (x in 0 until w) {
                val o = x * bpp
                px[y * w + x] = (0xFF shl 24) or ((cur[o].toInt() and 0xFF) shl 16) or ((cur[o + 1].toInt() and 0xFF) shl 8) or (cur[o + 2].toInt() and 0xFF)
            }
            System.arraycopy(cur, 0, prev, 0, stride)
        }
        return RgbFrame(w, h, px)
    }
}
