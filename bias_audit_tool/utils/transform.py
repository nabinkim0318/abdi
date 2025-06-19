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

        try:
            method = rec.split(" + ")[0]

            # Handle missing values
            if "DropHighNaNs" in rec:
                df_processed = df_processed[df_processed[col].notna()]
                if show_logs:
                    st.warning(f"🗑️ Dropped rows with NaNs in `{col}`")

            elif "ImputeMissing" in rec:
                fill_val = (
                    df_processed[col].mode()[0]
                    if df_processed[col].dtype == "object"
                    else df_processed[col].median()
                )
                df_processed[col] = (
                    df_processed[col].fillna(fill_val).infer_objects(copy=False)
                )
                if show_logs:
                    st.info(f"🧩 Imputed missing in `{col}` with `{fill_val}`")

            # Apply transformation
            if method == "OneHotEncoder":
                unique_count = df_processed[col].nunique()
                if unique_count > 50:
                    if show_logs:
                        st.warning(
                            f"⚠️ `{col}` has {unique_count} unique values. "
                            f"Skipped OneHotEncoder."
                        )
                    continue
                dummies = pd.get_dummies(df_processed[col], prefix=col)
                df_processed.drop(columns=[col], inplace=True)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                if show_logs:
                    st.success(
                        f"✅ `{col}` encoded (OneHot, {dummies.shape[1]} columns)."
                    )

            elif method == "LabelEncoder":
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

            elif method == "MinMaxScaler":
                scaler = MinMaxScaler()
                df_processed[col] = scaler.fit_transform(df_processed[[col]])
                if show_logs:
                    st.success(f"✅ `{col}` scaled (MinMaxScaler).")

            elif method == "Log1pTransform":
                if (df_processed[col] <= -1).any():
                    if show_logs:
                        st.warning(
                            f"⚠️ Skipping log1p for `{col}` " f"due to values ≤ -1."
                        )
                    continue
                df_processed[col] = np.log1p(df_processed[col])
                if show_logs:
                    st.info(f"🔁 Applied log1p transform to `{col}`")

            elif method == "DropColumn":
                df_processed.drop(columns=[col], inplace=True)
                if show_logs:
                    st.info(f"🗑️ `{col}` dropped.")

            elif method == "MeanImpute":
                fill_val = df_processed[col].mean()
                df_processed[col] = (
                    df_processed[col].fillna(fill_val).infer_objects(copy=False)
                )
                if show_logs:
                    st.success(f"✅ `{col}` imputed with mean: {fill_val:.2f}")

            elif method == "ModeImpute":
                fill_val = df_processed[col].mode()[0]
                df_processed[col] = (
                    df_processed[col].fillna(fill_val).infer_objects(copy=False)
                )
                if show_logs:
                    st.success(f"✅ `{col}` imputed with mode: {fill_val}")

            else:
                if show_logs:
                    st.warning(f"⚠️ Unknown method `{method}` for `{col}`")

        except Exception as e:
            if show_logs:
                st.error(f"❌ `{col}` failed with `{method}`: {e}")

    return df_processed
