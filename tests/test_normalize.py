"""
Unit tests for the unified normalize module.
Tests: country normalization (open-set), record normalization (preserving raw fields),
null address detection (has_address flag), and DataFrame batch normalization.
"""

import unittest
import pandas as pd
from src.business_entity_resolution.normalize import (
    normalize_country,
    normalize_dataframe,
    normalize_record,
)


class TestNormalize(unittest.TestCase):
    def test_normalize_country_open_set(self):
        # Open-set: should preserve countries absent from train (e.g. France, Germany)
        self.assertEqual(normalize_country("US"), "US")
        self.assertEqual(normalize_country("India"), "India")
        self.assertEqual(normalize_country("France"), "France")
        self.assertEqual(normalize_country("  France  "), "France")
        self.assertEqual(normalize_country("Deutschland"), "Deutschland")
        self.assertEqual(normalize_country(None), "")
        self.assertEqual(normalize_country(""), "")

    def test_normalize_record_preserves_raw_data(self):
        raw_record = {
            "entity_id": "S1-925783039",
            "business_name": "Orelee's Barbershop Inc.",
            "business_address": "1795 Westchester Drive, High Point, NC",
            "country": "US",
        }
        normalized = normalize_record(raw_record)
        
        # Verify raw keys are untouched
        self.assertEqual(normalized["entity_id"], "S1-925783039")
        self.assertEqual(normalized["business_name"], "Orelee's Barbershop Inc.")
        self.assertEqual(normalized["business_address"], "1795 Westchester Drive, High Point, NC")
        self.assertEqual(normalized["country"], "US")
        
        # Verify normalized keys are added
        self.assertEqual(normalized["normalized_name"], "orelee s barbershop inc")
        self.assertEqual(normalized["name_stem"], "orelee s barbershop")
        self.assertEqual(normalized["normalized_address"], "1795 westchester dr high point nc")
        self.assertTrue(normalized["has_address"])
        self.assertEqual(normalized["normalized_country"], "US")

    def test_normalize_record_null_address_fallback_flag(self):
        # Audit finding: ~3% of S2/S3 records have null address
        null_addr_record = {
            "entity_id": "S3-860443364",
            "business_name": "Maure Williams Inc Center",
            "business_address": None,
            "country": "US",
        }
        normalized = normalize_record(null_addr_record)
        self.assertEqual(normalized["normalized_address"], "")
        self.assertFalse(normalized["has_address"])
        self.assertEqual(normalized["normalized_name"], "maure williams inc center")

    def test_normalize_dataframe(self):
        data = {
            "entity_id": ["S1-1", "S2-2", "S3-3"],
            "business_name": ["Alpha Corp", "Beta Pvt Ltd", "Gamma LLC"],
            "business_address": ["123 Main Street", None, "456 Oak Avenue, Apt 2"],
            "country": ["US", "India", "France"],
        }
        df = pd.DataFrame(data)
        norm_df = normalize_dataframe(df)
        
        # Original columns preserved
        for col in ["entity_id", "business_name", "business_address", "country"]:
            self.assertIn(col, norm_df.columns)
            self.assertEqual(norm_df[col].tolist(), df[col].tolist())
        
        # New columns present
        self.assertIn("normalized_name", norm_df.columns)
        self.assertIn("name_stem", norm_df.columns)
        self.assertIn("normalized_address", norm_df.columns)
        self.assertIn("has_address", norm_df.columns)
        self.assertIn("normalized_country", norm_df.columns)
        
        # Check values
        self.assertEqual(norm_df["normalized_name"].tolist(), ["alpha corp", "beta pvt ltd", "gamma llc"])
        self.assertEqual(norm_df["name_stem"].tolist(), ["alpha", "beta", "gamma"])
        self.assertEqual(norm_df["normalized_address"].tolist(), ["123 main st", "", "456 oak ave apt 2"])
        self.assertEqual(norm_df["has_address"].tolist(), [True, False, True])
        self.assertEqual(norm_df["normalized_country"].tolist(), ["US", "India", "France"])


if __name__ == "__main__":
    unittest.main()
