import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def clean_label(col):
    return col.split(".")[-1].replace("_", " ").title()


def is_id_column(series):
    return series.nunique() >= 0.9 * len(series) and series.dtype == "object"


def is_categorical(series, threshold=10):
    return series.nunique() <= threshold or series.dtype == "object"


def reconstruct_demographic_groups(df, demographic_columns):
    """
    Convert binary encoded demographic columns back to meaningful groups
    """
    demographic_groups = {}
    
    # Group related columns
    column_groups = {
        'Gender': [col for col in demographic_columns if 'gender' in col.lower()],
        'Race': [col for col in demographic_columns if 'race' in col.lower()],
        'Ethnicity': [col for col in demographic_columns if 'ethnicity' in col.lower()],
        'Age Group': [col for col in demographic_columns if 'age' in col.lower() and 'obfuscated' not in col.lower()],
        'Disease Type': [col for col in demographic_columns if 'disease_type' in col.lower()]
    }
    
    for group_name, columns in column_groups.items():
        if not columns:
            continue
            
        # Skip if these look like continuous variables
        if group_name == 'Age Group' and any(df[col].dtype in ['int64', 'float64'] for col in columns if col in df.columns):
            continue
            
        # Reconstruct categorical data from binary columns
        group_data = []
        for idx in df.index:
            active_columns = []
            for col in columns:
                if col in df.columns and df.loc[idx, col] == 1:
                    # Extract category name from column name
                    category = col.split('_')[-1] if '_' in col else col.split('.')[-1]
                    active_columns.append(category)
            
            if active_columns:
                group_data.append(active_columns[0])  # Take first active category
            else:
                group_data.append('Unknown')
        
        if group_data and len(set(group_data)) > 1:  # Only add if we have meaningful variation
            demographic_groups[group_name] = pd.Series(group_data, index=df.index, name=group_name)
    
    return demographic_groups


def create_enhanced_demographic_charts(df, demographic_columns):
    """
    Create enhanced demographic visualizations with meaningful counts
    """
    st.subheader("📊 Enhanced Demographic Analysis")
    
    # Reconstruct demographic groups
    demo_groups = reconstruct_demographic_groups(df, demographic_columns)
    
    if not demo_groups:
        st.warning("⚠️ No meaningful demographic groups detected. Check your column selection.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Distribution Charts", "🎯 Bias Analysis", "📋 Summary Statistics"])
    
    with tab1:
        st.markdown("### Demographic Distribution Charts")
        
        for group_name, group_data in demo_groups.items():
            st.markdown(f"#### {group_name} Distribution")
            
            # Calculate counts and percentages
            counts = group_data.value_counts()
            total = len(group_data)
            
            # Create two columns for side-by-side charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart with counts
                fig_bar = px.bar(
                    x=counts.index,
                    y=counts.values,
                    title=f"{group_name} Counts (n={total})",
                    labels={'x': group_name, 'y': 'Count'},
                    color=counts.values,
                    color_continuous_scale='viridis'
                )
                fig_bar.update_layout(showlegend=False, height=400)
                
                # Add count labels on bars
                for i, (category, count) in enumerate(counts.items()):
                    fig_bar.add_annotation(
                        x=i, y=count + max(counts.values) * 0.01,
                        text=f"{count}<br>({count/total*100:.1f}%)",
                        showarrow=False,
                        font=dict(size=10)
                    )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Pie chart with percentages
                fig_pie = px.pie(
                    values=counts.values,
                    names=counts.index,
                    title=f"{group_name} Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont_size=10
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Summary statistics
            largest_group = counts.idxmax()
            largest_percentage = (counts.max() / total) * 100
            
            st.info(f"""
            **{group_name} Summary:**
            - Total participants: {total:,}
            - Largest group: {largest_group} ({counts.max():,} participants, {largest_percentage:.1f}%)
            - Number of categories: {len(counts)}
            - Diversity Index: {calculate_diversity_index(counts):.3f}
            """)
            
            st.markdown("---")
    
    with tab2:
        st.markdown("### Bias Detection Analysis")
        
        bias_results = []
        for group_name, group_data in demo_groups.items():
            counts = group_data.value_counts()
            bias_score = calculate_representation_bias(counts)
            
            # Determine bias level
            if bias_score > 0.6:
                bias_level = "High"
                color = "🔴"
            elif bias_score > 0.3:
                bias_level = "Medium"
                color = "🟡"
            else:
                bias_level = "Low"
                color = "🟢"
            
            bias_results.append({
                'Demographic Group': group_name,
                'Bias Score': f"{bias_score:.3f}",
                'Bias Level': f"{color} {bias_level}",
                'Largest Group %': f"{(counts.max() / counts.sum() * 100):.1f}%",
                'Total Participants': counts.sum()
            })
        
        # Display bias analysis table
        bias_df = pd.DataFrame(bias_results)
        st.dataframe(bias_df, use_container_width=True)
        
        # Bias alerts
        high_bias_groups = [result['Demographic Group'] for result in bias_results 
                           if 'High' in result['Bias Level']]
        
        if high_bias_groups:
            st.error(f"⚠️ **High bias detected** in: {', '.join(high_bias_groups)}")
            st.markdown("""
            **Recommendations:**
            - Review recruitment strategies for underrepresented groups
            - Consider stratified sampling approaches
            - Evaluate data collection methods for potential bias sources
            """)
        else:
            st.success("✅ No high bias detected in demographic representation")
    
    with tab3:
        st.markdown("### Detailed Statistics")
        
        for group_name, group_data in demo_groups.items():
            with st.expander(f"📈 {group_name} Detailed Statistics"):
                counts = group_data.value_counts()
                percentages = (counts / counts.sum() * 100).round(2)
                
                stats_df = pd.DataFrame({
                    'Category': counts.index,
                    'Count': counts.values,
                    'Percentage': percentages.values,
                    'Cumulative %': percentages.cumsum().values
                })
                
                st.dataframe(stats_df, use_container_width=True)
                
                # Download button for this group's data
                csv = stats_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {group_name} Data",
                    data=csv,
                    file_name=f"{group_name.lower().replace(' ', '_')}_statistics.csv",
                    mime="text/csv"
                )


def calculate_diversity_index(counts):
    """
    Calculate Simpson's Diversity Index (1 - sum of (p_i)^2)
    Higher values indicate more diversity
    """
    total = counts.sum()
    proportions = counts / total
    simpson_index = 1 - sum(proportions ** 2)
    return simpson_index


def calculate_representation_bias(counts):
    """
    Calculate representation bias score (0 = no bias, 1 = maximum bias)
    """
    total = counts.sum()
    expected_proportion = 1.0 / len(counts)
    actual_proportions = counts / total
    
    # Calculate bias as deviation from equal representation
    bias_score = sum(abs(prop - expected_proportion) for prop in actual_proportions) / 2
    return bias_score


def show_visualizations(df, audit_cols):
    """
    Enhanced visualization function that handles both binary and categorical columns
    """
    sns.set_theme(style="whitegrid", palette="pastel")
    audit_cols = [col for col in audit_cols if not is_id_column(df[col])]

    if not audit_cols:
        st.warning("⚠️ No meaningful columns to visualize. ID-like columns were removed.")
        return

    # Check if we have demographic columns for enhanced analysis
    demographic_cols = [col for col in audit_cols if any(keyword in col.lower() 
                       for keyword in ['demographic', 'gender', 'race', 'ethnicity', 'age'])]
    
    if demographic_cols:
        # Use enhanced demographic visualization
        create_enhanced_demographic_charts(df, demographic_cols)
    else:
        # Fall back to original visualization for non-demographic data
        show_traditional_visualizations(df, audit_cols)


def show_traditional_visualizations(df, audit_cols):
    """
    Traditional visualization approach for non-demographic columns
    """
    for col in audit_cols:
        if df[col].dropna().empty:
            st.warning(f"⚠️ Column `{col}` has only NaNs.")
            continue

        st.markdown(f"#### 🔍 Analysis for `{clean_label(col)}`")

        # Handle categorical columns
        if is_categorical(df[col]):
            df[col] = df[col].astype(str)
            fig, ax = plt.subplots(figsize=(10, 4))
            
            value_counts = df[col].value_counts()
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette="viridis")
            
            label = clean_label(col)
            ax.set_title(f"Distribution: {label}", fontsize=14)
            ax.set_xlabel("Count", fontsize=12)
            ax.set_ylabel(label, fontsize=12)
            
            # Add count labels
            for i, v in enumerate(value_counts.values):
                ax.text(v + max(value_counts.values) * 0.01, i, str(v), 
                       va='center', fontsize=10)
            
            st.pyplot(fig)
            plt.close(fig)

        # Handle numerical columns
        else:
            label = clean_label(col)
            col1, col2 = st.columns(2)
            
            with col1:
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                sns.histplot(df[col].dropna(), kde=True, ax=ax1, color="#4C72B0")
                ax1.set_title(f"Distribution: {label}", fontsize=12)
                ax1.set_xlabel(label, fontsize=10)
                st.pyplot(fig1)
                plt.close(fig1)

            with col2:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.boxplot(y=df[col].dropna(), ax=ax2, color="#55A868")
                ax2.set_title(f"Boxplot: {label}", fontsize=12)
                st.pyplot(fig2)
                plt.close(fig2)


def show_groupwise_visualizations(df, demo_cols, target_col=None):
    """
    Enhanced group-wise visualizations with better demographic handling
    """
    sns.set_theme(style="whitegrid", palette="pastel")
    
    # Filter to meaningful demographic columns
    meaningful_demo_cols = []
    for col in demo_cols:
        if col in df.columns and not is_id_column(df[col]):
            # Check if it's a meaningful demographic column
            if any(keyword in col.lower() for keyword in ['demographic', 'gender', 'race', 'ethnicity']):
                meaningful_demo_cols.append(col)

    if not meaningful_demo_cols:
        st.warning("⚠️ No meaningful demographic columns found for group-wise analysis.")
        return

    # First, try to reconstruct demographic groups
    demo_groups = reconstruct_demographic_groups(df, meaningful_demo_cols)
    
    if demo_groups:
        st.markdown("### 👥 Enhanced Group-wise Analysis")
        
        for group_name, group_data in demo_groups.items():
            st.markdown(f"#### Distribution by {group_name}")
            
            if target_col and target_col in df.columns:
                # Analysis with target variable
                df_combined = pd.DataFrame({
                    group_name: group_data,
                    target_col: df[target_col]
                })
                
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    # Boxplot for numeric target
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=df_combined, x=group_name, y=target_col, ax=ax)
                    ax.set_title(f"{clean_label(target_col)} by {group_name}")
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    # Stacked bar for categorical target
                    crosstab = pd.crosstab(group_data, df[target_col], normalize='index')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
                    ax.set_title(f"{clean_label(target_col)} Distribution by {group_name}")
                    ax.set_ylabel("Proportion")
                    ax.tick_params(axis='x', rotation=45)
                    ax.legend(title=clean_label(target_col), bbox_to_anchor=(1.05, 1), loc='upper left')
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                # Simple distribution without target
                counts = group_data.value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="viridis")
                ax.set_title(f"{group_name} Distribution")
                ax.set_ylabel("Count")
                ax.tick_params(axis='x', rotation=45)
                
                # Add count labels
                for i, v in enumerate(counts.values):
                    ax.text(i, v + max(counts.values) * 0.01, str(v), 
                           ha='center', va='bottom')
                
                st.pyplot(fig)
                plt.close(fig)
    else:
        # Fallback to original approach
        show_traditional_groupwise_visualizations(df, meaningful_demo_cols, target_col)


def show_traditional_groupwise_visualizations(df, demo_cols, target_col=None):
    """
    Original group-wise visualization approach (fallback)
    """
    for col in demo_cols:
        if col not in df.columns:
            continue
            
        st.markdown(f"#### 👥 Group-wise Distribution by `{clean_label(col)}`")

        df_plot = df.copy()
        if df[col].nunique() > 20:
            top_k = df[col].value_counts().nlargest(20).index
            df_plot = df[df[col].isin(top_k)]

        if pd.api.types.is_numeric_dtype(df_plot[col]) and df_plot[col].nunique() > 20:
            df_plot["__binned__"] = pd.cut(
                df_plot[col],
                bins=[0, 20, 40, 60, 80, 100, 120],
                labels=["0-20", "21-40", "41-60", "61-80", "81-100", "100+"],
            )
            x_col = "__binned__"
        else:
            x_col = col

        if target_col and target_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.boxplot(x=x_col, y=target_col, data=df_plot, ax=ax)
                ax.set_title(f"{clean_label(target_col)} by {clean_label(col)}")
                ax.set_xlabel(clean_label(col))
                ax.set_ylabel(clean_label(target_col))
            else:
                prop_df = (
                    df_plot.groupby([x_col, target_col])
                    .size()
                    .groupby(level=0)
                    .apply(lambda x: x / x.sum())
                    .reset_index(name="proportion")
                )
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=prop_df, x=x_col, y="proportion", hue=target_col, ax=ax)
                ax.set_title(f"{clean_label(target_col)} Proportion per {clean_label(col)}")
                ax.set_xlabel(clean_label(col))
                ax.set_ylabel("Proportion")
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            if df_plot[col].dtype == "object" or df_plot[col].nunique() < 10:
                sns.countplot(x=x_col, data=df_plot, ax=ax)
                ax.set_ylabel("Count")
            else:
                sns.histplot(x=df_plot[col], ax=ax, bins=20)
                ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {clean_label(col)}")
            ax.set_xlabel(clean_label(col))

        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# Keep all your existing helper functions unchanged
def show_demographic_overview(df, demographic_cols):
    sns.set_theme(style="whitegrid", palette="pastel")
    num_cols = len(demographic_cols)
    fig, axs = plt.subplots(nrows=num_cols, ncols=2, figsize=(12, 4 * num_cols))

    for i, col in enumerate(demographic_cols):
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col].dropna(), ax=axs[i, 0], kde=True)
            axs[i, 0].set_title(f"Histogram of {clean_label(col)}")
            sns.boxplot(x=df[col].dropna(), ax=axs[i, 1], color="#55A868")
            axs[i, 1].set_title(f"Boxplot of {clean_label(col)}")
        else:
            sns.countplot(
                x=col, data=df, ax=axs[i, 0], order=df[col].value_counts().index
            )
            axs[i, 0].set_title(f"Value Counts of {clean_label(col)}")
            axs[i, 0].tick_params(axis="x", rotation=30)
            axs[i, 1].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def restore_group_column(df, prefix, new_col_name):
    """
    Restore one-hot encoded columns to group columns.
    """
    onehot_cols = [col for col in df.columns if col.startswith(prefix)]
    if not onehot_cols:
        raise ValueError(f"No columns found with prefix: {prefix}")

    def get_group(row):
        for col in onehot_cols:
            if row[col] == 1:
                return col.replace(prefix, "")
        return "Unknown"

    df_copy = df.copy()
    df_copy[new_col_name] = df_copy[onehot_cols].apply(get_group, axis=1)
    return df_copy


def auto_group_selector(df, merge_all=True):
    """
    Either merge all one-hot columns automatically or allow manual selection
    """
    def extract_prefixes(df):
        return sorted(
            {
                ".".join(col.split("_")[0:2])
                for col in df.columns
                if "_" in col and len(col.split("_")) >= 2
            }
        )

    st.sidebar.markdown("### 🔄 Auto One-Hot Column Restoration")
    prefixes = extract_prefixes(df)

    if not prefixes:
        st.sidebar.info("No One-hot Encoded Columns Found.")
        return df, None

    st.sidebar.write(f"Found {len(prefixes)} one-hot encoded column groups:")
    for prefix in prefixes[:3]:
        st.sidebar.write(f"• {prefix}_...")

    if "restored_df" in st.session_state and "restored_col" in st.session_state:
        st.sidebar.success(
            f"✅ Using previously restored column: {st.session_state['restored_col']}"
        )
        return st.session_state["restored_df"], st.session_state["restored_col"]

    selected_prefix = st.sidebar.selectbox(
        "Select One-hot Prefix to Restore",
        prefixes,
        key="prefix_selector_unique",
    )

    if selected_prefix:
        onehot_cols = [
            col for col in df.columns if col.startswith(selected_prefix + "_")
        ]
        st.sidebar.markdown(f"**Preview:** {len(onehot_cols)} columns found:")
        for col in onehot_cols[:3]:
            st.sidebar.markdown(f"• `{col}`")
        if len(onehot_cols) > 3:
            st.sidebar.markdown(f"• ... and {len(onehot_cols) - 3} more")

    new_col_name = st.sidebar.text_input(
        "New Column Name", value="group", key="col_name_input_unique"
    )

    if st.sidebar.button("Restore Group Column", key="restore_button_unique"):
        try:
            df_new = restore_group_column(df, selected_prefix + "_", new_col_name)
            st.session_state["restored_df"] = df_new
            st.session_state["restored_col"] = new_col_name
            st.session_state["restored_prefix"] = selected_prefix
            st.sidebar.success(f"✅ Restored Column: {new_col_name}")
            return df_new, new_col_name
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            return df, None

    return df, None


def plot_radar_chart(df, group_col, metrics):
    try:
        valid_metrics = [metric for metric in metrics if metric in df.columns]

        if not valid_metrics:
            st.warning("No valid metrics found for radar chart.")
            return

        if len(valid_metrics) < 2:
            st.warning("Need at least 2 metrics for radar chart.")
            return

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[valid_metrics])
        df_scaled = pd.DataFrame(scaled_data, columns=valid_metrics)
        df_scaled[group_col] = df[group_col].values

        avg_by_group = df_scaled.groupby(group_col).mean()

        categories = valid_metrics
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for group, row in avg_by_group.iterrows():
            values = row.tolist()
            if len(values) == N:
                values.append(values[0])
            elif len(values) != N + 1:
                continue
            if len(angles) != len(values):
                continue
            ax.plot(angles, values, label=group)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title("Demographic Group Comparison", fontsize=16)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Error creating radar chart: {str(e)}")


def plot_distribution_comparison(result_df, top_n=20):
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = result_df.index.astype(str)

    ax.bar(idx, result_df["Observed_%"], width=0.4, label="Observed", align="edge")
    ax.bar(idx, result_df["Expected_%"], width=-0.4, label="Expected", align="edge")

    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels(idx, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Observed vs Expected Group Distribution")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig
