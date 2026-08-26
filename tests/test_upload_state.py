import io

import pandas as pd
import pytest

from bias_audit_tool.data.upload_state import apply_upload_identity
from bias_audit_tool.data.upload_state import begin_new_upload
from bias_audit_tool.data.upload_state import clear_dataset_if_upload_removed
from bias_audit_tool.data.upload_state import DATASET_DERIVED_SESSION_KEYS
from bias_audit_tool.data.upload_state import DEFAULT_UPLOADER_WIDGET_KEY
from bias_audit_tool.data.upload_state import reset_dataset_state
from bias_audit_tool.data.upload_state import UPLOADER_SESSION_KEY
from bias_audit_tool.data.validation import fingerprint_upload


def _session_with_analysis_state():
    return {
        "df": pd.DataFrame({"x": [1, 2]}),
        "df_proc": pd.DataFrame({"x": [1]}),
        "recommendations": {"x": "MinMaxScaler"},
        "preprocessing_applied": True,
        "step3_ready": True,
        "demo_cols": ["group"],
        "group_col": "group",
        "group_col_selectbox": "group",
        "last_group_col": "group",
        "target_col": "y",
        "trigger_audit": True,
        "audit_run_id": "old-id",
        "audit_error_msg": "stale",
        "fairness_result": {"dp": 0.1},
        "step3_done": True,
        "show_visualization": True,
        "upload_fingerprint": "aaa",
        "uploaded_file_name": "A.csv",
    }


def test_same_bytes_do_not_reset_dataset_state():
    session = _session_with_analysis_state()
    payload = b"a,b\n1,2\n"
    digest = fingerprint_upload(io.BytesIO(payload))
    session["upload_fingerprint"] = digest
    reset = apply_upload_identity(session, filename="A.csv", fingerprint=digest)
    assert reset is False
    assert session["df"] is not None
    assert session["preprocessing_applied"] is True
    assert session["target_col"] == "y"
    assert session["group_col"] == "group"
    assert session["uploaded_file_name"] == "A.csv"


def test_same_filename_changed_bytes_resets_dataset_state():
    session = _session_with_analysis_state()
    old = fingerprint_upload(io.BytesIO(b"a,b\n1,2\n"))
    new = fingerprint_upload(io.BytesIO(b"a,b\n3,4\n"))
    assert old != new
    session["upload_fingerprint"] = old
    session["uploaded_file_name"] = "A.csv"
    reset = apply_upload_identity(session, filename="A.csv", fingerprint=new)
    assert reset is True
    assert session["df"] is None
    assert session["df_proc"] is None
    assert session["recommendations"] is None
    assert session["preprocessing_applied"] is False
    assert session["step3_ready"] is False
    assert session["target_col"] is None
    assert "demo_cols" not in session
    assert "group_col" not in session
    assert "group_col_selectbox" not in session
    assert "last_group_col" not in session
    assert session["upload_fingerprint"] == new
    assert session["uploaded_file_name"] == "A.csv"
    # Display preference is not a dataset-derived key.
    assert session["show_visualization"] is True


def test_different_filename_same_bytes_preserves_analytical_identity():
    session = _session_with_analysis_state()
    payload = b"a,b\n1,2\n"
    digest = fingerprint_upload(io.BytesIO(payload))
    session["upload_fingerprint"] = digest
    session["uploaded_file_name"] = "A.csv"
    reset = apply_upload_identity(session, filename="B.csv", fingerprint=digest)
    assert reset is False
    assert session["df"] is not None
    assert session["target_col"] == "y"
    assert session["uploaded_file_name"] == "B.csv"


def test_fingerprint_does_not_consume_stream():
    raw = b"col1,col2\n10,20\n30,40\n"
    buffer = io.BytesIO(raw)
    fingerprint_upload(buffer)
    df = pd.read_csv(buffer)
    assert list(df.columns) == ["col1", "col2"]
    assert len(df) == 2


def test_reset_dataset_state_clears_authoritative_keys():
    session = _session_with_analysis_state()
    reset_dataset_state(session)
    for key in DATASET_DERIVED_SESSION_KEYS:
        if key in ("df", "df_proc", "recommendations", "target_col"):
            assert session.get(key) in (None,)
        elif key in ("preprocessing_applied", "step3_ready", "trigger_audit"):
            assert session.get(key) is False
    assert "group_col_selectbox" not in session
    assert session["show_visualization"] is True


def test_apptest_same_filename_replacement_is_not_simulated():
    """AppTest cannot reliably replace UploadedFile bytes in this Streamlit version.

    Session identity is tested through ``apply_upload_identity``, which is the
    function ``app.py`` calls on every upload. Pretending a source-code grep
    covers the stale-state bug would be insufficient; the helper is the
    authoritative state transition.
    """
    pytest.importorskip("streamlit")
    session = _session_with_analysis_state()
    apply_upload_identity(session, filename="A.csv", fingerprint="new-digest")
    assert session["df"] is None
    assert session["preprocessing_applied"] is False


def test_cleared_uploader_resets_dataset_derived_state():
    session = _session_with_analysis_state()
    cleared = clear_dataset_if_upload_removed(session, uploaded_file=None)
    assert cleared is True
    assert session["df"] is None
    assert session["df_proc"] is None
    assert session["preprocessing_applied"] is False
    assert session["target_col"] is None
    assert session.get("upload_fingerprint") is None
    assert session.get("uploaded_file_name") is None
    assert session["show_visualization"] is True


def test_cleared_uploader_without_prior_identity_does_not_reset():
    session = _session_with_analysis_state()
    session.pop("upload_fingerprint")
    df = session["df"]
    cleared = clear_dataset_if_upload_removed(session, uploaded_file=None)
    assert cleared is False
    assert session["df"] is df
    assert session["target_col"] == "y"


def test_present_upload_does_not_clear_on_rerun():
    session = _session_with_analysis_state()
    cleared = clear_dataset_if_upload_removed(session, uploaded_file=object())
    assert cleared is False
    assert session["df"] is not None
    assert session["upload_fingerprint"] == "aaa"


def test_try_another_dataset_rotates_uploader_key_and_drops_identity():
    session = _session_with_analysis_state()
    session[UPLOADER_SESSION_KEY] = DEFAULT_UPLOADER_WIDGET_KEY
    previous_key = session[UPLOADER_SESSION_KEY]
    new_key = begin_new_upload(session)
    assert new_key != previous_key
    assert session[UPLOADER_SESSION_KEY] == new_key
    assert session.get("upload_fingerprint") is None
    assert session["df"] is None
    assert session["preprocessing_applied"] is False
    # Previous widget key is no longer the active uploader identity, so
    # Streamlit will not immediately rebind the last uploaded bytes.
    assert new_key != DEFAULT_UPLOADER_WIDGET_KEY
