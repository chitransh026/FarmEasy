import numpy as np
import cv2
from PIL import Image
import io

class FieldHealthAnalyzer:

    def load_image(self, image_file):
        """Load image and ensure RGB format"""
        img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        return np.array(img)

    # ---------- CHANNEL EXTRACTION ----------

    def extract_channels(self, img):
        r = img[:, :, 0].astype(float)
        g = img[:, :, 1].astype(float)
        b = img[:, :, 2].astype(float)
        return r, g, b

    def simulate_nir(self, r, g, b):
        """Simulated NIR band (for RGB-only imagery)"""
        return 0.6 * r + 0.3 * g + 0.1 * b

    # ---------- VEGETATION INDICES ----------

    def calculate_ndvi(self, r, nir):
        ndvi = (nir - r) / (nir + r + 1e-10)
        return np.clip(ndvi, -1, 1)

    def calculate_ndwi(self, g, nir):
        ndwi = (g - nir) / (g + nir + 1e-10)
        return np.clip(ndwi, -1, 1)

    # ---------- STRESS MASKS ----------

    def vegetation_mask(self, ndvi):
        """Healthy vegetation detection"""
        return ndvi > 0.3

    def water_stress_mask(self, ndwi):
        """Water stress detection"""
        return ndwi < -0.2

    def pest_stress_mask(self, img):
        """
        Pest / disease stress proxy:
        Detects yellowing and browning using HSV color space
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        yellow_mask = cv2.inRange(
            hsv, (20, 80, 80), (35, 255, 255)
        )

        brown_mask = cv2.inRange(
            hsv, (10, 100, 20), (20, 255, 200)
        )

        return (yellow_mask | brown_mask) > 0

    # ---------- RECOMMENDATIONS ----------

    def get_crop_specific_recommendations(
        self, ndvi_mean, water_stress_percent, crop_name=None
    ):
        """
        Rule-based agronomic recommendations.
        crop_name is optional and treated as contextual text only.
        """

        recommendations = []

        # Vegetation condition
        if ndvi_mean < 0.3:
            recommendations.append("Apply nitrogen-rich fertilizer")
            recommendations.append("Inspect field for pest or disease stress")

        # Water condition
        if water_stress_percent > 60:
            recommendations.append("Immediate irrigation required")
            recommendations.append("Consider drip irrigation for efficiency")
        elif water_stress_percent > 40:
            recommendations.append("Schedule irrigation within the next 3 days")

        # Optimal condition
        if ndvi_mean > 0.6 and water_stress_percent < 30:
            recommendations.append("Optimal growth conditions detected")
            recommendations.append("Maintain current crop management practices")

        # Optional crop context (no database)
        if crop_name and ndvi_mean > 0.5:
            recommendations.append(
                f"{crop_name.capitalize()}: Support current growth stage with balanced nutrients"
            )

        return recommendations

    # ---------- HEALTH ASSESSMENT ----------

    def get_health_assessment(self, ndvi, water_stress_mask, crop_name=None):
        """Generate overall field health summary"""

        mean_ndvi = float(np.mean(ndvi))
        water_stress_percent = float(np.mean(water_stress_mask) * 100)

        # Vegetation health scoring
        if mean_ndvi > 0.6:
            health = "Excellent"
            health_score = 90
        elif mean_ndvi > 0.3:
            health = "Good"
            health_score = 70
        elif mean_ndvi > 0.1:
            health = "Moderate"
            health_score = 50
        else:
            health = "Poor"
            health_score = 30

        # Water condition scoring
        if water_stress_percent < 30:
            water_status = "Adequate moisture"
            water_score = 80
        elif water_stress_percent < 60:
            water_status = "Moderate water stress"
            water_score = 50
        else:
            water_status = "High water stress"
            water_score = 20

        # Recommendations
        recommendations = self.get_crop_specific_recommendations(
            mean_ndvi, water_stress_percent, crop_name
        )

        overall_score = int(health_score * 0.6 + water_score * 0.4)

        return {
            "overall_health": health,
            "health_score": health_score,
            "water_status": water_status,
            "water_score": water_score,
            "overall_score": overall_score,
            "recommendations": recommendations,
            "summary": (
                f"Crop health is {health.lower()} "
                f"(NDVI: {mean_ndvi:.2f}). "
                f"Water status: {water_status.lower()} "
                f"(Stress: {water_stress_percent:.1f}%)."
            ),
            "indices_summary": {
                "ndvi_mean": mean_ndvi,
                "water_stress_percent": water_stress_percent
            }
        }

    # ---------- MAIN PIPELINE ----------

    def analyze(self, image_file, crop_name=None):
        """Run full field health analysis"""

        img = self.load_image(image_file)

        r, g, b = self.extract_channels(img)
        nir = self.simulate_nir(r, g, b)

        ndvi = self.calculate_ndvi(r, nir)
        ndwi = self.calculate_ndwi(g, nir)

        veg_mask = self.vegetation_mask(ndvi)
        water_stress = self.water_stress_mask(ndwi)
        pest_stress = self.pest_stress_mask(img)

        health_assessment = self.get_health_assessment(
            ndvi, water_stress, crop_name
        )

        return {
            "vegetation_coverage_percent": float(np.mean(veg_mask) * 100),
            "water_stress_percent": float(np.mean(water_stress) * 100),
            "pest_stress_percent": float(np.mean(pest_stress & veg_mask) * 100),
            "health_assessment": health_assessment
        }
