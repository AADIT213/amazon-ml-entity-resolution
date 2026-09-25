"""
Unit tests for text_utils module.
Tests: whitespace, diacritics, Unicode to ASCII, punctuation stripping, and null safety.
"""

import math
import unittest
from src.business_entity_resolution.text_utils import (
    clean_whitespace,
    is_null_or_empty,
    remove_diacritics,
    strip_punctuation,
    to_ascii,
)


class TestTextUtils(unittest.TestCase):
    def test_is_null_or_empty(self):
        self.assertTrue(is_null_or_empty(None))
        self.assertTrue(is_null_or_empty(float("nan")))
        self.assertTrue(is_null_or_empty(math.nan))
        self.assertTrue(is_null_or_empty(""))
        self.assertTrue(is_null_or_empty("   "))
        self.assertTrue(is_null_or_empty("\t\n"))
        self.assertFalse(is_null_or_empty("a"))
        self.assertFalse(is_null_or_empty(123))

    def test_clean_whitespace(self):
        self.assertEqual(clean_whitespace(None), "")
        self.assertEqual(clean_whitespace(""), "")
        self.assertEqual(clean_whitespace("   hello   world   "), "hello world")
        self.assertEqual(clean_whitespace("multiple \t\n whitespace \r\n characters"), "multiple whitespace characters")
        self.assertEqual(clean_whitespace("non\u00a0breaking\u2000space"), "non breaking space")

    def test_remove_diacritics(self):
        self.assertEqual(remove_diacritics(None), "")
        self.assertEqual(remove_diacritics("Café Société Résumé"), "Cafe Societe Resume")
        self.assertEqual(remove_diacritics("München Über"), "Munchen Uber")
        self.assertEqual(remove_diacritics("Château d'Hôtel"), "Chateau d'Hotel")
        self.assertEqual(remove_diacritics("Zürich"), "Zurich")
        self.assertEqual(remove_diacritics("François"), "Francois")

    def test_to_ascii(self):
        self.assertEqual(to_ascii(None), "")
        # Curly quotes
        self.assertEqual(to_ascii("‘single’ and “double”"), "'single' and \"double\"")
        # En and em dashes
        self.assertEqual(to_ascii("word–dash—word"), "word-dash-word")
        # Ampersand expansion
        self.assertEqual(to_ascii("Johnson & Johnson"), "Johnson and Johnson")
        self.assertEqual(to_ascii("B&M"), "B and M")
        # Slashes replaced by space to prevent token merge
        self.assertEqual(to_ascii("A/B"), "A B")
        # German double s ligature
        self.assertEqual(to_ascii("Straße"), "Strasse")

    def test_indic_transliteration(self):
        # Local rule-based transliteration of Devanagari script per TRD §3
        self.assertEqual(
            to_ascii("होटल एंटरप्राइजेज लिमिटेड"),
            "hotl entrpraijej limited",
        )

    def test_strip_punctuation(self):
        self.assertEqual(strip_punctuation(None), "")
        self.assertEqual(strip_punctuation("Hello, World!"), "Hello World")
        self.assertEqual(strip_punctuation("B+ Retail Inc."), "B Retail Inc")
        self.assertEqual(strip_punctuation("Payne-Enterprises"), "Payne Enterprises")
        self.assertEqual(strip_punctuation("Entity [Co.]"), "Entity Co")

    def test_idempotence(self):
        sample = "  Château & Co., 1056–1060 Belden Ave.  "
        first = to_ascii(sample)
        second = to_ascii(first)
        self.assertEqual(first, second)

        first_clean = clean_whitespace(sample)
        self.assertEqual(first_clean, clean_whitespace(first_clean))


if __name__ == "__main__":
    unittest.main()
