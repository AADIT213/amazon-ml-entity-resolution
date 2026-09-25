"""
Unified Entity Normalization Interface.

Combines name, address, and country normalizers into an idempotent,
pure transformation pipeline that preserves all raw input fields.
"""

from typing import Any, Dict, Optional
import pandas as pd
from .address_norm import normalize_address
from .name_norm import normalize_name, strip_legal_suffix
from .text_utils import clean_whitespace, is_null_or_empty, to_ascii


def normalize_country(country: Optional[str]) -> str:
    """
    Normalize country string without hard-coding an enum.
    Country is open-set (contains France in test set, US/India in train, etc.).
    Preserves raw identity while stripping outer whitespace and normalizing Unicode.
    """
    if is_null_or_empty(country):
        return ""
    return clean_whitespace(to_ascii(country))


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an entity record dictionary while preserving raw values.
    
    Inputs:
        record: dict containing raw keys (e.g. 'entity_id', 'business_name',
                'business_address', 'country').
    
    Returns:
        new dict with all original keys preserved intact, plus:
        - 'normalized_name': standardized business name
        - 'name_stem': business name with legal suffix stripped
        - 'normalized_address': standardized business address
        - 'has_address': boolean indicating non-empty normalized address (for null fallback)
        - 'normalized_country': cleaned country string (open-set)
    """
    raw_name = record.get("business_name")
    raw_addr = record.get("business_address")
    raw_ctry = record.get("country")
    
    norm_name = normalize_name(raw_name)
    stem_name = strip_legal_suffix(norm_name)
    norm_addr = normalize_address(raw_addr)
    norm_ctry = normalize_country(raw_ctry)
    
    result = dict(record)
    result["normalized_name"] = norm_name
    result["name_stem"] = stem_name
    result["normalized_address"] = norm_addr
    result["has_address"] = bool(norm_addr)
    result["normalized_country"] = norm_ctry
    
    return result


def normalize_dataframe(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Apply deterministic normalization to a pandas DataFrame.
    Preserves original columns and appends normalized columns.
    
    New columns added:
    - 'normalized_name'
    - 'name_stem'
    - 'normalized_address'
    - 'has_address'
    - 'normalized_country'
    """
    if not inplace:
        df = df.copy()
    
    if "business_name" in df.columns:
        df["normalized_name"] = df["business_name"].fillna("").astype(str).map(normalize_name)
        df["name_stem"] = df["normalized_name"].map(strip_legal_suffix)
    
    if "business_address" in df.columns:
        df["normalized_address"] = df["business_address"].fillna("").astype(str).map(normalize_address)
        df["has_address"] = df["normalized_address"] != ""
    
    if "country" in df.columns:
        df["normalized_country"] = df["country"].fillna("").astype(str).map(normalize_country)
    
    return df
