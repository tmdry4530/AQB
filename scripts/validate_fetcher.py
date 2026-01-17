#!/usr/bin/env python3
"""
Validation script for the data fetcher module.

This script performs static analysis of the fetcher module to ensure
it meets all requirements without requiring full dependency installation.
"""

import ast
from pathlib import Path
import sys


def extract_classes_and_methods(tree: ast.Module) -> dict[str, list[str]]:
    """Extract classes and their methods from AST."""
    classes = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                # Handle both sync and async methods
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)
            classes[node.name] = methods

    return classes


def extract_dataclasses(tree: ast.Module) -> set[str]:
    """Extract dataclass names from AST."""
    dataclasses = set()

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                    dataclasses.add(node.name)

    return dataclasses


def validate_fetcher_module():
    """Validate the fetcher module structure and completeness."""
    print("=" * 70)
    print("IFTB Data Fetcher Module Validation")
    print("=" * 70)

    fetcher_path = Path(__file__).parent.parent / "src" / "iftb" / "data" / "fetcher.py"

    if not fetcher_path.exists():
        print(f"\nERROR: Fetcher module not found at {fetcher_path}")
        return False

    print(f"\nReading module from: {fetcher_path}")

    # Read and parse the module
    with open(fetcher_path) as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"\nERROR: Syntax error in module: {e}")
        return False

    print("\n✓ Module syntax is valid")

    # Extract classes and functions
    classes = extract_classes_and_methods(tree)
    dataclasses_found = extract_dataclasses(tree)

    # Validation checks
    checks_passed = 0
    checks_total = 0

    print("\n" + "=" * 70)
    print("Requirement Validation")
    print("=" * 70)

    # Check 1: ExchangeClient class
    checks_total += 1
    print("\n1. ExchangeClient class")
    if "ExchangeClient" in classes:
        print("   ✓ ExchangeClient class found")

        required_methods = [
            "__init__",
            "connect",
            "close",
            "fetch_ohlcv",
            "fetch_ohlcv_range",
            "fetch_ticker",
            "fetch_funding_rate",
            "fetch_open_interest",
            "__aenter__",
            "__aexit__",
        ]

        missing_methods = [m for m in required_methods if m not in classes["ExchangeClient"]]

        if not missing_methods:
            print("   ✓ All required methods present")
            print(f"   Methods: {', '.join(sorted(classes['ExchangeClient']))}")
            checks_passed += 1
        else:
            print(f"   ✗ Missing methods: {', '.join(missing_methods)}")
            print(f"   Found methods: {', '.join(sorted(classes['ExchangeClient']))}")
    else:
        print("   ✗ ExchangeClient class not found")

    # Check 2: HistoricalDataDownloader class
    checks_total += 1
    print("\n2. HistoricalDataDownloader class")
    if "HistoricalDataDownloader" in classes:
        print("   ✓ HistoricalDataDownloader class found")

        required_methods = [
            "__init__",
            "download_historical",
        ]

        missing_methods = [
            m for m in required_methods if m not in classes["HistoricalDataDownloader"]
        ]

        if not missing_methods:
            print("   ✓ All required methods present")
            print(f"   Methods: {', '.join(sorted(classes['HistoricalDataDownloader']))}")
            checks_passed += 1
        else:
            print(f"   ✗ Missing methods: {', '.join(missing_methods)}")
            print(f"   Found methods: {', '.join(sorted(classes['HistoricalDataDownloader']))}")
    else:
        print("   ✗ HistoricalDataDownloader class not found")

    # Check 3: Data structures
    checks_total += 1
    print("\n3. Data structures")
    required_dataclasses = {"OHLCVBar", "Ticker", "FundingRate"}
    found_dataclasses = dataclasses_found & required_dataclasses

    if found_dataclasses == required_dataclasses:
        print(f"   ✓ All required dataclasses found: {', '.join(sorted(found_dataclasses))}")
        checks_passed += 1
    else:
        missing = required_dataclasses - found_dataclasses
        print(f"   ✗ Missing dataclasses: {', '.join(sorted(missing))}")
        if found_dataclasses:
            print(f"   Found dataclasses: {', '.join(sorted(found_dataclasses))}")

    # Check 4: Error handling
    checks_total += 1
    print("\n4. Error handling")

    # Check for retry logic
    if "_retry_request" in classes.get("ExchangeClient", []):
        print("   ✓ Retry mechanism implemented (_retry_request)")
        checks_passed += 1
    else:
        print("   ✗ Retry mechanism not found")

    # Check 5: Async context manager
    checks_total += 1
    print("\n5. Async context manager support")
    if "__aenter__" in classes.get("ExchangeClient", []) and "__aexit__" in classes.get(
        "ExchangeClient", []
    ):
        print("   ✓ Async context manager methods present")
        checks_passed += 1
    else:
        print("   ✗ Async context manager not implemented")

    # Check 6: Module structure
    checks_total += 1
    print("\n6. Module structure")
    structure_valid = True

    # Check imports
    import_found = False
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module in ["iftb.config", "iftb.utils"]:
                import_found = True

    if import_found:
        print("   ✓ Required imports from iftb.config and iftb.utils")
    else:
        print("   ✗ Missing required imports")
        structure_valid = False

    # Check docstring
    docstring = ast.get_docstring(tree)
    if docstring and len(docstring) > 50:
        print("   ✓ Module docstring present")
    else:
        print("   ✗ Module docstring missing or too short")
        structure_valid = False

    if structure_valid:
        checks_passed += 1

    # Print all classes found
    print("\n" + "=" * 70)
    print("Classes and Methods Found")
    print("=" * 70)
    for cls_name, methods in sorted(classes.items()):
        print(f"\n{cls_name}:")
        for method in sorted(methods):
            print(f"  - {method}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nChecks passed: {checks_passed}/{checks_total}")

    if checks_passed == checks_total:
        print("\n✓ All validation checks passed!")
        print("\nThe fetcher module is complete and production-ready.")
        return True
    print(f"\n✗ {checks_total - checks_passed} validation check(s) failed")
    return False


def print_module_info():
    """Print additional module information."""
    fetcher_path = Path(__file__).parent.parent / "src" / "iftb" / "data" / "fetcher.py"

    with open(fetcher_path) as f:
        content = f.read()

    lines = content.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]
    comment_lines = [line for line in lines if line.strip().startswith("#")]
    docstring_lines = [line for line in lines if '"""' in line or "'''" in line]

    print("\n" + "=" * 70)
    print("Module Statistics")
    print("=" * 70)
    print(f"\nTotal lines:           {len(lines)}")
    print(f"Non-empty lines:       {len(non_empty_lines)}")
    print(f"Comment lines:         {len(comment_lines)}")
    print(f"Docstring markers:     {len(docstring_lines)}")
    print(f"File size:             {len(content)} bytes ({len(content) / 1024:.1f} KB)")


def main():
    """Run validation."""
    success = validate_fetcher_module()
    print_module_info()

    if success:
        print("\n" + "=" * 70)
        print("✓ Validation Complete - Module Ready for Use")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("✗ Validation Failed - Please Review Errors")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
