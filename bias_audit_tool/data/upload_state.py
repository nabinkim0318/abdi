# Dataset identity and Streamlit session reset for uploaded CSVs.
#
# Analytical session state is keyed by a SHA-256 content fingerprint, not by
# filename. Byte-identical uploads keep analysis state even if the filename
# changes; a new digest resets dataset-derived keys. The digest is stored in
# session state only — it is not logged or transmitted.
import uuid
from typing import MutableMapping
from typing import Optional


# Authoritative list of keys derived from the uploaded dataset / analysis.
# User-facing display preferences such as ``show_visualization`` are not
# included.
DATASET_DERIVED_SESSION_KEYS = (
    "df",
    "df_proc",
    "recommendations",
    "preprocessing_applied",
    "step3_ready",
    "demo_cols",
    "group_col",
    "last_group_col",
    "target_col",
    "trigger_audit",
    "audit_run_id",
    "audit_error_msg",
    "audit_error_trace",
    "fairness_result",
    "step3_done",
    "structural_issues",
)

# Streamlit widget keys that would otherwise restore a stale selection.
DATASET_WIDGET_KEYS = ("group_col_selectbox",)

DATASET_STATE_DEFAULTS = {
    "df": None,
    "df_proc": None,
    "recommendations": None,
    "preprocessing_applied": False,
    "step3_ready": False,
    "target_col": None,
    "trigger_audit": False,
}

UPLOADER_SESSION_KEY = "csv_uploader_key"
DEFAULT_UPLOADER_WIDGET_KEY = "csv_uploader"


def reset_dataset_state(
    session_state: MutableMapping,
    *,
    clear_identity: bool = False,
) -> None:
    """Clear dataset-derived analysis state.

    ``upload_fingerprint`` is kept unless ``clear_identity`` is True (used
    by “try another dataset”). Filename display metadata is updated by the
    caller after identity is known.
    """
    for key in DATASET_DERIVED_SESSION_KEYS:
        if key in DATASET_STATE_DEFAULTS:
            session_state[key] = DATASET_STATE_DEFAULTS[key]
        else:
            session_state.pop(key, None)
    for key in DATASET_WIDGET_KEYS:
        session_state.pop(key, None)
    session_state["audit_run_id"] = uuid.uuid4()
    if clear_identity:
        session_state.pop("upload_fingerprint", None)
        session_state.pop("uploaded_file_name", None)


def apply_upload_identity(
    session_state: MutableMapping,
    *,
    filename: str,
    fingerprint: str,
) -> bool:
    """Bind the current upload to session state.

    Returns True when dataset-derived state was reset because the content
    fingerprint changed (including the first upload). Same-byte reruns
    return False and only refresh ``uploaded_file_name`` for display.
    """
    previous: Optional[str] = session_state.get("upload_fingerprint")
    if previous != fingerprint:
        reset_dataset_state(session_state, clear_identity=False)
        session_state["upload_fingerprint"] = fingerprint
        session_state["uploaded_file_name"] = filename
        return True
    session_state["uploaded_file_name"] = filename
    return False


def uploader_widget_key(session_state: MutableMapping) -> str:
    """Stable Streamlit file_uploader key until a new-upload reset."""
    key = session_state.get(UPLOADER_SESSION_KEY)
    if not key:
        session_state[UPLOADER_SESSION_KEY] = DEFAULT_UPLOADER_WIDGET_KEY
    return session_state[UPLOADER_SESSION_KEY]


def clear_dataset_if_upload_removed(
    session_state: MutableMapping,
    uploaded_file,
) -> bool:
    """Clear analysis state when the uploader is emptied after a prior upload.

    Ordinary reruns with no upload identity are left alone. A previous
    ``upload_fingerprint`` plus a now-empty widget is treated as a removed
    dataset, not as a reason to keep analyzing the last DataFrame.
    """
    if uploaded_file is not None:
        return False
    if session_state.get("upload_fingerprint") is None:
        return False
    reset_dataset_state(session_state, clear_identity=True)
    return True


def begin_new_upload(session_state: MutableMapping) -> str:
    """Drop dataset state and rotate the file_uploader widget key.

    Rotating the key prevents Streamlit from immediately reusing the
    previous uploaded bytes after “Try with another dataset?”.
    """
    reset_dataset_state(session_state, clear_identity=True)
    session_state[UPLOADER_SESSION_KEY] = (
        f"{DEFAULT_UPLOADER_WIDGET_KEY}_{uuid.uuid4().hex}"
    )
    return session_state[UPLOADER_SESSION_KEY]
