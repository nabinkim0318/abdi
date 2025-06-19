import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def apply_preprocessing(
    df: pd.DataFrame, recommendations: dict, show_logs: bool = True
) -> pd.DataFrame:
    df_processed = df.copy()
    if show_logs:
        st.markdown("### 🛠️ Preprocessing Log")

    for col, rec in recommendations.items():
        if col not in df_processed.columns:
            if show_logs:
                st.warning(f"⚠️ `{col}` not found. Skipping.")
            continue

        methods = rec.split(" + ")

        # 🔹 Drop entire column
        if "DropColumn" in methods:
            df_processed.drop(columns=[col], inplace=True)
            if show_logs:
                st.info(f"🗑️ `{col}` dropped (DropColumn).")
            continue

        # 🔹 Drop rows with too many NaNs
        if "DropHighNaNs" in methods:
            df_processed.drop(columns=[col], inplace=True)
            if show_logs:
                st.warning(
                    f"🗑️ `{col}` dropped due to ≥95% missing values "
                    "(DropHighNaNs)."
                )
            continue

        # 🔹 Imputation
        if any("Impute" in m for m in methods):
            if df_processed[col].dtype == "object":
                mode_series = df_processed[col].mode(dropna=True)
                if not mode_series.empty:
                    fill_val = mode_series[0]
                else:
                    fill_val = "missing_fallback"
                    if show_logs:
                        st.warning(
                            f"⚠️ `{col}` has no mode. Using fallback value: "
                            f"'{fill_val}'"
                        )
            else:
                fill_val = df_processed[col].median()
            df_processed[col] = (
                df_processed[col].fillna(fill_val).infer_objects(copy=False)
            )
            if show_logs:
                st.info(f"🧩 Imputed missing in `{col}` with `{fill_val}`.")

        # 🔹 Encoding and Transformations
        try:
            if "OneHotEncoder" in methods:
                unique_count = df_processed[col].nunique()
                if unique_count > 50:
                    if show_logs:
                        st.warning(
                            f"⚠️ `{col}` has {unique_count} unique values. "
                            "Skipped OneHotEncoder."
                        )
                    continue
                dummies = pd.get_dummies(df_processed[col], prefix=col)
                df_processed.drop(columns=[col], inplace=True)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                if show_logs:
                    st.success(
                        f"✅ `{col}` encoded (OneHot, {dummies.shape[1]} columns)."
                    )

            elif "LabelEncoder" in methods:
                df_processed[col] = (
                    df_processed[col].fillna("missing_label").astype(str)
                )
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                if show_logs:
                    st.success(
                        f"✅ `{col}` encoded with LabelEncoder "
                        f"(missing → 'missing_label')."
                    )

            elif "MinMaxScaler" in methods:
                scaler = MinMaxScaler()
                df_processed[col] = scaler.fit_transform(df_processed[[col]])
                if show_logs:
                    st.success(f"✅ `{col}` scaled (MinMaxScaler).")

            elif "Log1pTransform" in methods:
                if (df_processed[col] <= -1).any():
                    if show_logs:
                        st.warning(
                            f"⚠️ Skipping log1p for `{col}` due to values ≤ -1."
                        )
                else:
                    df_processed[col] = np.log1p(df_processed[col])
                    if show_logs:
                        st.info(f"🔁 Applied log1p transform to `{col}`.")

        except Exception as e:
            if show_logs:
                st.error(f"❌ `{col}` failed with preprocessing step: {e}")

    return df_processed
