def show_visualizations(df, audit_cols):
    sns.set_theme(style="whitegrid", palette="pastel")
    audit_cols = [col for col in audit_cols if col in df.columns and not is_id_column(df[col])]

    if not audit_cols:
        st.warning("⚠️ No meaningful columns to visualize. ID-like columns were removed.")
        return

    # Dummy-based one‑hot groups (kept)
    gender_cols = ["demographic.gender_female", "demographic.gender_male"]
    race_cols = [
        "demographic.race_white",
        "demographic.race_black or african american",
        "demographic.race_asian",
        "demographic.race_american indian or alaska native",
        "demographic.race_not reported",
    ]
    ethnicity_cols = [
        "demographic.ethnicity_hispanic or latino",
        "demographic.ethnicity_not reported",
    ]

    demographic_columns, other_columns = [], []
    for col in audit_cols:
        if col in gender_cols + race_cols + ethnicity_cols or "demographic" in col.lower():
            demographic_columns.append(col)
        else:
            other_columns.append(col)

    # Include age explicitly if present (belt and suspenders)
    if "demographic.age_at_index" in df.columns and "demographic.age_at_index" not in demographic_columns:
        demographic_columns.append("demographic.age_at_index")

    if demographic_columns:
        st.markdown("#### 🔍 Demographic Distributions")

        # filter out all‑NaN
        valid_demo_cols = [c for c in demographic_columns if c in df.columns and not df[c].dropna().empty]
        if not valid_demo_cols:
            st.warning("⚠️ All demographic columns are empty.")
            return

        n_cols = min(3, len(valid_demo_cols))
        n_rows = (len(valid_demo_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for i, col in enumerate(valid_demo_cols):
            ax = axes[i]
            series = df[col]

            # Numeric demographic (e.g., age)
            if pd.api.types.is_numeric_dtype(series):
                # Histogram
                sns.histplot(series.dropna(), bins=20, kde=True, ax=ax)
                ax.set_title(f"{clean_label(col)} (Histogram)", fontsize=12)
                ax.set_xlabel(clean_label(col), fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
            else:
                # Categorical / one‑hot -> countplot
                temp = series.astype(str)
                sns.countplot(x=temp, ax=ax, order=temp.value_counts().index)
                ax.set_title(f"{clean_label(col)}", fontsize=12)
                ax.set_xlabel(clean_label(col), fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                ax.tick_params(axis="x", rotation=45, labelsize=9)

        for j in range(len(valid_demo_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
