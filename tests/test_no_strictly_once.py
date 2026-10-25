from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_no_strictly_once_phrase():
    """Ensures the misleading phrase 'strictly once' does not appear anywhere in README, docs, or test reports."""
    target_files = [REPO_ROOT / "README.md"]
    target_files.extend((REPO_ROOT / "docs").glob("**/*.md"))
    target_files.extend((REPO_ROOT / "results").glob("**/*.md"))
    target_files.extend((REPO_ROOT / "results").glob("**/*.json"))
    target_files.extend((REPO_ROOT / "results").glob("**/*.txt"))

    forbidden = "strictly once"
    violations = []

    for file_path in target_files:
        if not file_path.exists():
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if forbidden in content.lower():
            violations.append(str(file_path.relative_to(REPO_ROOT)))

    assert not violations, f"Forbidden phrase '{forbidden}' found in: {violations}"

def test_verbatim_test_use_protocol_statement():
    """Validates that the honest test-use protocol statement is present in README and MANUSCRIPT_REVISION."""
    expected_fragment = (
        "The detector hyperparameters, threshold and persistence rule were frozen and committed "
        "(frozen_config.json, SHA-256 recorded) before the first test evaluation. "
        "The official test set was therefore used once for the detector. "
        "Downstream triage/trust components were developed after the test labels had been visible; "
        "their results are reported as exploratory unless their parameters were selected on "
        "Dataset 04 validation data only, as documented in"
    )

    readme_content = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    manuscript_content = (REPO_ROOT / "docs" / "MANUSCRIPT_REVISION.md").read_text(encoding="utf-8")

    # Normalize backticks and whitespaces for comparison
    def clean(text: str) -> str:
        return " ".join(text.replace("`", "").split())

    clean_expected = clean(expected_fragment)
    assert clean_expected in clean(readme_content), "Honest test-use statement missing or modified in README.md"
    assert clean_expected in clean(manuscript_content), "Honest test-use statement missing or modified in MANUSCRIPT_REVISION.md"
