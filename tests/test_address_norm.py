"""
Unit tests for address_norm module.
Tests: casing, thoroughfare abbreviations, unit/floor abbreviations,
numeric token normalization, punctuation, null handling, and idempotence.
"""

import unittest
from src.business_entity_resolution.address_norm import normalize_address


class TestAddressNorm(unittest.TestCase):
    def test_null_and_empty(self):
        self.assertEqual(normalize_address(None), "")
        self.assertEqual(normalize_address(""), "")
        self.assertEqual(normalize_address("   "), "")
        self.assertEqual(normalize_address(float("nan")), "")

    def test_casing_and_whitespace(self):
        self.assertEqual(
            normalize_address("1795 WESTCHESTER DRIVE, HIGH POINT, NC"),
            "1795 westchester dr high point nc",
        )
        self.assertEqual(
            normalize_address("  17560   Ellis   Road,   Tahlequah,   OK  "),
            "17560 ellis rd tahlequah ok",
        )

    def test_thoroughfare_abbreviations(self):
        self.assertEqual(
            normalize_address("3315 Fremont Street, Peoria, IL"),
            "3315 fremont st peoria il",
        )
        self.assertEqual(
            normalize_address("1056 Belden Avenue, Akron, OH"),
            "1056 belden ave akron oh",
        )
        self.assertEqual(
            normalize_address("1056-1060 BELDEN AVE, AKRON, OH"),
            "1056 1060 belden ave akron oh",
        )
        self.assertEqual(
            normalize_address("5559 Orville Boulevard, Columbus, OH"),
            "5559 orville blvd columbus oh",
        )
        self.assertEqual(
            normalize_address("700 Industrial Parkway"),
            "700 industrial pkwy",
        )

    def test_unit_and_floor_designations(self):
        self.assertEqual(
            normalize_address("2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD"),
            "2100 cameron dr unit apt g dundalk md",
        )
        self.assertEqual(
            normalize_address("728 A Quail Avenue, Fl Ground Floor, Geneva, IA"),
            "728 a quail ave fl ground fl geneva ia",
        )
        self.assertEqual(
            normalize_address("PO BOX 8807, Akron, OH"),
            "po box 8807 akron oh",
        )
        self.assertEqual(
            normalize_address("P.O. Box 1234, Suite 500"),
            "po box 1234 ste 500",
        )

    def test_numeric_token_normalization(self):
        # Ordinals: 11th -> 11, 14th -> 14, 1st -> 1
        self.assertEqual(
            normalize_address("11Th Floor, N1 Block Embassy Tech Park, Bangalore"),
            "11 fl n1 block embassy tech park bangalore",
        )
        self.assertEqual(
            normalize_address("1401, 14Th Floor, Plot -161"),
            "1401 14 fl plot 161",
        )
        # Numbers: #162 -> no 162
        self.assertEqual(
            normalize_address("#162 14th Floor"),
            "no 162 14 fl",
        )
        self.assertEqual(
            normalize_address("WZ-187C Shop No.13, Delhi"),
            "wz 187c shop no 13 delhi",
        )

    def test_duplicate_token_cleanup(self):
        # 'Unit Unit 2' -> 'unit 2'
        self.assertEqual(
            normalize_address("294 Meadowcreek Drive, Unit Unit 2, Pewaukee, WI"),
            "294 meadowcreek dr unit 2 pewaukee wi",
        )

    def test_french_address_terms(self):
        # Open-set France test data support
        self.assertEqual(
            normalize_address("12 Rue de l'Hôtel de Ville, 75004 Paris"),
            "12 rue de l hotel de ville 75004 paris",
        )
        self.assertEqual(
            normalize_address("45 Boulevard Saint-Germain, 3ème Étage"),
            "45 blvd st germain 3eme fl",
        )

    def test_idempotence(self):
        samples = [
            "1795 Westchester Drive, High Point, NC",
            "2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD",
            "1401, 14Th Floor, Plot -161, Raheja Majestic, Mumbai",
            "1056-1060 BELDEN AVE, PO BOX 8807, AKRON, OH",
            "12 Rue de l'Hôtel de Ville, 75004 Paris",
        ]
        for s in samples:
            norm1 = normalize_address(s)
            norm2 = normalize_address(norm1)
            self.assertEqual(norm1, norm2, f"Failed address idempotence for: {s}")


if __name__ == "__main__":
    unittest.main()
