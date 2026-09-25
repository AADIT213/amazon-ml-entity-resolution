"""
Unit tests for name_norm module.
Tests: casing, whitespace, punctuation, legal suffix standardization,
stem extraction, null handling, and idempotence.
"""

import unittest
from src.business_entity_resolution.name_norm import normalize_name, strip_legal_suffix


class TestNameNorm(unittest.TestCase):
    def test_null_and_empty(self):
        self.assertEqual(normalize_name(None), "")
        self.assertEqual(normalize_name(""), "")
        self.assertEqual(normalize_name("   "), "")
        self.assertEqual(strip_legal_suffix(None), "")
        self.assertEqual(strip_legal_suffix(""), "")

    def test_casing_and_whitespace(self):
        self.assertEqual(normalize_name("ORELEE'S BARBERSHOP"), "orelee s barbershop")
        self.assertEqual(normalize_name("   prime    money   "), "prime money")
        self.assertEqual(normalize_name("MoOrE   BiTwIsE"), "moore bitwise")

    def test_punctuation_handling(self):
        # Hyphens replaced by spaces to prevent token concatenation
        self.assertEqual(normalize_name("PAYNE-ENTERPRISES"), "payne enterprises")
        self.assertEqual(normalize_name("B+ Retail Inc"), "b retail inc")
        self.assertEqual(normalize_name("True Factory-Ltd"), "true factory ltd")
        self.assertEqual(normalize_name("ZANDER BLUE [CO]"), "zander blue co")
        self.assertEqual(normalize_name("Hendricks & Flowers Inc."), "hendricks and flowers inc")

    def test_indian_legal_suffixes(self):
        self.assertEqual(
            normalize_name("Consulting Nyasa Nursing Private Limited"),
            "consulting nyasa nursing pvt ltd",
        )
        self.assertEqual(
            normalize_name("Systel Buildstructure (India) Private Ltd"),
            "systel buildstructure india pvt ltd",
        )
        self.assertEqual(
            normalize_name("Alpha Engineering Pvt. Ltd."),
            "alpha engineering pvt ltd",
        )
        self.assertEqual(
            normalize_name("Beta Traders P. Ltd."),
            "beta traders pvt ltd",
        )
        self.assertEqual(
            normalize_name("Gamma Tech Limited"),
            "gamma tech ltd",
        )

    def test_us_legal_suffixes(self):
        self.assertEqual(normalize_name("Custom Wealth Services LLC"), "custom wealth services llc")
        self.assertEqual(normalize_name("Custom Wealth Services L.L.C."), "custom wealth services llc")
        self.assertEqual(normalize_name("Moore Bitwise Inc"), "moore bitwise inc")
        self.assertEqual(normalize_name("Moore Bitwise Incorporated"), "moore bitwise inc")
        self.assertEqual(normalize_name("Universal Corporation"), "universal corp")
        self.assertEqual(normalize_name("Zander Blue Company"), "zander blue co")
        # Ensure 'company' in the middle of a name is NOT converted
        self.assertEqual(normalize_name("Company of Wolves"), "company of wolves")

    def test_european_french_legal_suffixes(self):
        # Open-set France support
        self.assertEqual(normalize_name("Boulangerie Dupont SARL"), "boulangerie dupont sarl")
        self.assertEqual(normalize_name("Société Anonyme Chrono SA"), "sa chrono sa")
        self.assertEqual(normalize_name("Tech Innovations SAS"), "tech innovations sas")
        self.assertEqual(normalize_name("Immobilier Du Centre SCI"), "immobilier du centre sci")
        self.assertEqual(normalize_name("Auto Handel GmbH"), "auto handel gmbh")

    def test_strip_legal_suffix(self):
        self.assertEqual(
            strip_legal_suffix("Custom Wealth Services LLC"),
            "custom wealth services",
        )
        self.assertEqual(
            strip_legal_suffix("Systel Buildstructure India Private Limited"),
            "systel buildstructure india",
        )
        self.assertEqual(
            strip_legal_suffix("Moore Bitwise Inc."),
            "moore bitwise",
        )
        # Leading legal suffix
        self.assertEqual(
            strip_legal_suffix("LLC Orellana Investments"),
            "orellana investments",
        )
        # Name that is only a legal suffix should not become empty
        self.assertEqual(strip_legal_suffix("LLC"), "llc")
        self.assertEqual(strip_legal_suffix("Inc"), "inc")

    def test_idempotence(self):
        samples = [
            "Orelee's Barbershop",
            "B+ Retail Inc.",
            "Systel Buildstructure (India) Private Limited",
            "Custom Wealth Services LLC",
            "PAYNE-ENTERPRISES",
            "ZANDER BLUE [CO]",
        ]
        for s in samples:
            norm1 = normalize_name(s)
            norm2 = normalize_name(norm1)
            self.assertEqual(norm1, norm2, f"Failed idempotence for: {s}")

            stem1 = strip_legal_suffix(s)
            stem2 = strip_legal_suffix(stem1)
            self.assertEqual(stem1, stem2, f"Failed stem idempotence for: {s}")


if __name__ == "__main__":
    unittest.main()
