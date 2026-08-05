import pytest

from src.evaluation.p4_exceedance.manifest import (
    build_manifest,
    REQUIRED_MANIFEST_FIELDS,
    PROJECT,
    ANALYSIS_ROLE,
)


class TestManifest:
    def test_all_required_fields_present(self):
        manifest = build_manifest()
        assert set(manifest.keys()) == set(REQUIRED_MANIFEST_FIELDS)

    def test_project_and_role_are_fixed_and_cannot_be_overridden(self):
        manifest = build_manifest(project="something else", analysis_role="something else")
        assert manifest["project"] == PROJECT
        assert manifest["analysis_role"] == ANALYSIS_ROLE
        assert "P4" in PROJECT

    def test_unknown_field_rejected(self):
        with pytest.raises(ValueError, match="Unknown manifest field"):
            build_manifest(not_a_real_field=1)

    def test_unknown_values_are_null_or_pending_not_fabricated(self):
        manifest = build_manifest()
        assert manifest["input_predictions_path"] is None
        assert manifest["station"] is None
        assert manifest["producer_repository"] == "PENDING_VERIFICATION"

    def test_provided_fields_override_defaults(self):
        manifest = build_manifest(station="elche", random_seed=42)
        assert manifest["station"] == "elche"
        assert manifest["random_seed"] == 42
