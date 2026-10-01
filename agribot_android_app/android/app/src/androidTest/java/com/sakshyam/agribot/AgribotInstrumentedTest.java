package com.sakshyam.agribot;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.res.XmlResourceParser;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.xmlpull.v1.XmlPullParser;

@RunWith(AndroidJUnit4.class)
public class AgribotInstrumentedTest {
    @Test
    public void usesNativeAgribotPackage() {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();

        assertEquals("com.sakshyam.agribot", appContext.getPackageName());
    }

    @Test
    public void productionManifestDoesNotRequestInternet() throws Exception {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        PackageInfo packageInfo = appContext.getPackageManager()
                .getPackageInfo(appContext.getPackageName(), PackageManager.GET_PERMISSIONS);
        String[] requestedPermissions = packageInfo.requestedPermissions == null
                ? new String[0]
                : packageInfo.requestedPermissions;

        assertFalse(Arrays.asList(requestedPermissions).contains(android.Manifest.permission.INTERNET));
        assertTrue(Arrays.asList(requestedPermissions).contains(android.Manifest.permission.CAMERA));
        assertTrue(Arrays.asList(requestedPermissions).contains(android.Manifest.permission.ACCESS_COARSE_LOCATION));
        assertTrue(Arrays.asList(requestedPermissions).contains(android.Manifest.permission.ACCESS_FINE_LOCATION));
    }

    @Test
    public void manifestDisablesBackupAndCleartext() throws Exception {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        ApplicationInfo applicationInfo = appContext.getPackageManager()
                .getApplicationInfo(appContext.getPackageName(), 0);

        assertFalse((applicationInfo.flags & ApplicationInfo.FLAG_ALLOW_BACKUP) != 0);
        assertFalse((applicationInfo.flags & ApplicationInfo.FLAG_USES_CLEARTEXT_TRAFFIC) != 0);
    }

    @Test
    public void fileProviderOnlySharesPrivateExports() throws Exception {
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        try (XmlResourceParser parser = appContext.getResources().getXml(R.xml.file_paths)) {
            int filesPathCount = 0;
            int eventType = parser.getEventType();
            while (eventType != XmlPullParser.END_DOCUMENT) {
                if (eventType == XmlPullParser.START_TAG) {
                    String tag = parser.getName();
                    assertFalse("external-path must not be exposed", "external-path".equals(tag));
                    assertFalse("cache-path must not be exposed", "cache-path".equals(tag));
                    if ("files-path".equals(tag)) {
                        filesPathCount++;
                        assertEquals("agribot_exports", parser.getAttributeValue(null, "name"));
                        assertEquals("exports/", parser.getAttributeValue(null, "path"));
                    }
                }
                eventType = parser.next();
            }
            assertEquals(1, filesPathCount);
        }
    }

}
