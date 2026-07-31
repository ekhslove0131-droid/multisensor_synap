from pathlib import Path


def test_publish_script_is_private_hash_gated_and_non_destructive() -> None:
    script = Path("scripts/publish_kaggle_hierarchical_model.sh").read_text()

    assert "kaggle models create" in script
    assert "kaggle models variations create" in script
    assert "kaggle models variations versions create" in script
    assert "kaggle models variations versions download" in script
    assert "multisensor-ml kaggle-model verify" in script
    assert "model_manifest.json" in script
    assert "archive_sha256" in script
    assert "row.get(\"versionNumber\", row.get(\"version\"))" in script
    assert "for attempt in {1..12}" in script
    assert "remote_variation_handle" in script
    assert "/ScikitLearn/oracle-sanity-v1" in script
    assert "--untar" in script
    assert script.index("versions list") < script.index("kaggle models create")
    assert "variations get" not in script
    assert "--untar -f -q >/dev/null" in script
    assert "KAGGLE_API_TOKEN" not in script
    assert "models delete" not in script
    assert "kernels push" not in script
