#!/usr/bin/env python3
"""
Validation script for LLM Market Analyzer implementation.

This script performs comprehensive checks to ensure the LLM analyzer
is properly implemented and ready for use.
"""

import sys
from pathlib import Path

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print section header."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")


def print_check(passed: bool, message: str) -> None:
    """Print check result."""
    symbol = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    print(f"{symbol} {message}")


def validate_file_exists(filepath: Path) -> bool:
    """Check if file exists."""
    return filepath.exists()


def validate_file_content(filepath: Path, required_strings: list[str]) -> tuple[bool, list[str]]:
    """Check if file contains required strings."""
    try:
        content = filepath.read_text()
        missing = [s for s in required_strings if s not in content]
        return len(missing) == 0, missing
    except Exception as e:
        return False, [f"Error reading file: {e}"]


def main() -> int:
    """Run validation checks."""
    project_root = Path(__file__).parent.parent
    src_root = project_root / "src" / "iftb"

    print(f"\n{BLUE}LLM Market Analyzer Validation{RESET}")
    print(f"Project: {project_root}")

    total_checks = 0
    passed_checks = 0

    # Check 1: Core Implementation File
    print_header("1. Core Implementation File")

    llm_analyzer_file = src_root / "analysis" / "llm_analyzer.py"
    exists = validate_file_exists(llm_analyzer_file)
    print_check(exists, f"File exists: {llm_analyzer_file}")
    total_checks += 1
    passed_checks += 1 if exists else 0

    if exists:
        required = [
            "class SentimentScore",
            "class FallbackMode",
            "class LLMAnalysis",
            "class LLMVetoSystem",
            "class LLMAnalyzer",
            "async def analyze_market",
            "def should_veto_trade",
            "async def create_analyzer_from_settings",
        ]
        passed, missing = validate_file_content(llm_analyzer_file, required)
        for item in required:
            has_item = item not in missing
            print_check(has_item, f"Contains: {item}")
            total_checks += 1
            passed_checks += 1 if has_item else 0

    # Check 2: Module Exports
    print_header("2. Module Exports")

    init_file = src_root / "analysis" / "__init__.py"
    exists = validate_file_exists(init_file)
    print_check(exists, f"File exists: {init_file}")
    total_checks += 1
    passed_checks += 1 if exists else 0

    if exists:
        required = [
            "LLMAnalyzer",
            "LLMVetoSystem",
            "LLMAnalysis",
            "SentimentScore",
            "FallbackMode",
            "create_analyzer_from_settings",
        ]
        passed, missing = validate_file_content(init_file, required)
        for item in required:
            has_item = item not in missing
            print_check(has_item, f"Exports: {item}")
            total_checks += 1
            passed_checks += 1 if has_item else 0

    # Check 3: Test File
    print_header("3. Test File")

    test_file = project_root / "tests" / "unit" / "test_llm_analyzer.py"
    exists = validate_file_exists(test_file)
    print_check(exists, f"File exists: {test_file}")
    total_checks += 1
    passed_checks += 1 if exists else 0

    if exists:
        required = [
            "class TestSentimentScore",
            "class TestLLMAnalysis",
            "class TestLLMVetoSystem",
            "class TestLLMAnalyzer",
            "def test_veto",
            "def test_sentiment",
            "def test_confidence",
        ]
        passed, missing = validate_file_content(test_file, required)
        test_count = content.count("def test_") if (content := test_file.read_text()) else 0
        print_check(test_count >= 20, f"Test count: {test_count} (expected: ≥20)")
        total_checks += 1
        passed_checks += 1 if test_count >= 20 else 0

    # Check 4: Documentation
    print_header("4. Documentation")

    docs = [
        ("Main Documentation", project_root / "docs" / "LLM_ANALYZER.md"),
        ("Quick Reference", project_root / "docs" / "LLM_ANALYZER_QUICK_REFERENCE.md"),
        ("Implementation Summary", project_root / "LLM_ANALYZER_IMPLEMENTATION_SUMMARY.md"),
    ]

    for name, filepath in docs:
        exists = validate_file_exists(filepath)
        print_check(exists, f"{name}: {filepath.name}")
        total_checks += 1
        passed_checks += 1 if exists else 0

    # Check 5: Example File
    print_header("5. Example File")

    example_file = project_root / "examples" / "llm_analyzer_example.py"
    exists = validate_file_exists(example_file)
    print_check(exists, f"File exists: {example_file}")
    total_checks += 1
    passed_checks += 1 if exists else 0

    if exists:
        required = [
            "async def example_basic_analysis",
            "async def example_veto_system",
            "async def example_with_caching",
            "async def example_health_check",
            "async def example_error_handling",
        ]
        passed, missing = validate_file_content(example_file, required)
        for item in required:
            has_item = item not in missing
            print_check(has_item, f"Contains: {item}")
            total_checks += 1
            passed_checks += 1 if has_item else 0

    # Check 6: Configuration Constants
    print_header("6. Configuration Constants")

    constants_file = src_root / "config" / "constants.py"
    exists = validate_file_exists(constants_file)
    print_check(exists, f"File exists: {constants_file}")
    total_checks += 1
    passed_checks += 1 if exists else 0

    if exists:
        required = [
            "SENTIMENT_VETO_THRESHOLD",
            "CONFIDENCE_VETO_THRESHOLD",
            "SENTIMENT_CAUTION_THRESHOLD",
            "CONFIDENCE_CAUTION_THRESHOLD",
            "NEWS_CONFLICT_PENALTY",
        ]
        passed, missing = validate_file_content(constants_file, required)
        for item in required:
            has_item = item not in missing
            print_check(has_item, f"Defines: {item}")
            total_checks += 1
            passed_checks += 1 if has_item else 0

    # Check 7: Settings Integration
    print_header("7. Settings Integration")

    settings_file = src_root / "config" / "settings.py"
    exists = validate_file_exists(settings_file)
    print_check(exists, f"File exists: {settings_file}")
    total_checks += 1
    passed_checks += 1 if exists else 0

    if exists:
        required = [
            "class LLMSettings",
            "anthropic_api_key",
            "model",
            "max_tokens",
            "cache_ttl_seconds",
        ]
        passed, missing = validate_file_content(settings_file, required)
        for item in required:
            has_item = item not in missing
            print_check(has_item, f"Contains: {item}")
            total_checks += 1
            passed_checks += 1 if has_item else 0

    # Check 8: Dependencies
    print_header("8. Dependencies")

    pyproject_file = project_root / "pyproject.toml"
    exists = validate_file_exists(pyproject_file)
    print_check(exists, f"File exists: {pyproject_file}")
    total_checks += 1
    passed_checks += 1 if exists else 0

    if exists:
        required = [
            "anthropic",
            "redis",
            "orjson",
            "httpx",
        ]
        passed, missing = validate_file_content(pyproject_file, required)
        for item in required:
            has_item = item not in missing
            print_check(has_item, f"Dependency: {item}")
            total_checks += 1
            passed_checks += 1 if has_item else 0

    # Check 9: Syntax Validation
    print_header("9. Syntax Validation")

    files_to_check = [
        ("Core Implementation", llm_analyzer_file),
        ("Test File", test_file),
        ("Example File", example_file),
    ]

    for name, filepath in files_to_check:
        if filepath.exists():
            try:
                import py_compile
                py_compile.compile(str(filepath), doraise=True)
                print_check(True, f"{name}: Syntax valid")
                passed_checks += 1
            except Exception as e:
                print_check(False, f"{name}: Syntax error - {e}")
        else:
            print_check(False, f"{name}: File not found")
        total_checks += 1

    # Final Summary
    print_header("Validation Summary")

    percentage = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    print(f"Total Checks: {total_checks}")
    print(f"Passed: {GREEN}{passed_checks}{RESET}")
    print(f"Failed: {RED}{total_checks - passed_checks}{RESET}")
    print(f"Success Rate: {GREEN if percentage == 100 else YELLOW}{percentage:.1f}%{RESET}")

    if percentage == 100:
        print(f"\n{GREEN}✓ All validation checks passed!{RESET}")
        print(f"{GREEN}  The LLM Market Analyzer is properly implemented.{RESET}\n")
        return 0
    elif percentage >= 90:
        print(f"\n{YELLOW}⚠ Most validation checks passed.{RESET}")
        print(f"{YELLOW}  Minor issues detected. Review failed checks.{RESET}\n")
        return 1
    else:
        print(f"\n{RED}✗ Validation failed.{RESET}")
        print(f"{RED}  Multiple issues detected. Review implementation.{RESET}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
