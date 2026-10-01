package com.sakshyam.agribot.featurescan

import com.sakshyam.agribot.featurescan.walk.parseInRange
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ResourcesAndInputsTest {
    private fun strings(path: String): Map<String, String> {
        val doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(File(path))
        val nodes = doc.getElementsByTagName("string")
        return (0 until nodes.length).associate { i -> nodes.item(i).attributes.getNamedItem("name").nodeValue to nodes.item(i).textContent }
    }

    private val en = strings("src/main/res/values/strings.xml")
    private val hi = strings("src/main/res/values-hi/strings.xml")
    private val formatArg = Regex("%(\\d+\\$)?[sd]")

    @Test fun `every English string has a Hindi translation with the same format arguments`() {
        val missing = en.keys - hi.keys
        assertTrue("missing Hindi strings: $missing", missing.isEmpty())
        assertTrue("Hindi-only strings: ${hi.keys - en.keys}", (hi.keys - en.keys).isEmpty())
        for ((k, v) in en) {
            assertEquals("format args differ for $k", formatArg.findAll(v).map { it.value }.toSortedSet(), formatArg.findAll(hi.getValue(k)).map { it.value }.toSortedSet())
        }
    }

    @Test fun `every model label has localised condition text`() {
        val manifest = Json.parseToJsonElement(File("../app/src/main/assets/model_manifest.json").readText()).jsonObject
        val labels = manifest["classifier"]!!.jsonObject["labels"]!!.jsonArray.map { it.jsonPrimitive.content }
        for (l in labels) assertTrue("no text for model label $l", Conditions.forKey(l) != null)
        assertEquals(labels.toSet(), Conditions.knownKeys)
    }

    @Test fun `numeric field parsing accepts commas and enforces ranges`() {
        assertEquals(0.6, parseInRange("0,6", 0.1, 5.0)!!, 1e-9)
        assertEquals(12.0, parseInRange(" 12 ", 1.0, 100.0)!!, 1e-9)
        assertNull(parseInRange("0", 1.0, 100.0))
        assertNull(parseInRange("abc", 1.0, 100.0))
        assertNull(parseInRange("", 1.0, 100.0))
    }
}
