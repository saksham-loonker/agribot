package com.sakshyam.agribot;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import javax.xml.XMLConstants;
import javax.xml.parsers.DocumentBuilderFactory;
import org.junit.Test;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

public class AgribotUnitTest {
    private static final String ANDROID_NS = "http://schemas.android.com/apk/res/android";

    @Test
    public void productionManifestStaysOfflineAndLocalOnly() throws Exception {
        Document manifest = parseManifest();

        NodeList permissionNodes = manifest.getElementsByTagName("uses-permission");
        List<String> permissions = new ArrayList<>();
        for (int index = 0; index < permissionNodes.getLength(); index++) {
            Element permission = (Element) permissionNodes.item(index);
            permissions.add(permission.getAttributeNS(ANDROID_NS, "name"));
        }

        assertEquals(List.of(
                "android.permission.CAMERA",
                "android.permission.ACCESS_COARSE_LOCATION",
                "android.permission.ACCESS_FINE_LOCATION",
                "android.permission.ACTIVITY_RECOGNITION"), permissions);
        assertFalse("android.permission.INTERNET must not be present", permissions.contains("android.permission.INTERNET"));

        Element application = (Element) manifest.getElementsByTagName("application").item(0);
        assertEquals("false", application.getAttributeNS(ANDROID_NS, "usesCleartextTraffic"));
        assertEquals("false", application.getAttributeNS(ANDROID_NS, "allowBackup"));
        assertEquals("false", application.getAttributeNS(ANDROID_NS, "fullBackupContent"));
    }

    private static Document parseManifest() throws Exception {
        File manifestFile = new File("src/main/AndroidManifest.xml");
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setNamespaceAware(true);
        factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);
        return factory.newDocumentBuilder().parse(manifestFile);
    }
}
