# Shiny app
from shiny import App, reactive, ui, render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats


# Load datasets using pandas
nhanes_dict = pd.read_csv("/Users/chancenlaw/Desktop/nhanes_dictionary.csv")
nhanes_biomarkers = pd.read_csv("/Users/chancenlaw/Desktop/nhanes_markers.csv")
nhanes_labels = pd.read_csv("/Users/chancenlaw/Desktop/nhanes_heart_disease_labels.csv")

# Map biomarker labels to their English text
label_mapping = dict(zip(nhanes_dict["Variable.Name"], nhanes_dict["Variable.Description"]))

# Preserve critical columns (SEQN and RIAGENDR) during renaming
label_mapping.pop("SEQN", None)  # Ensure SEQN is not renamed
label_mapping.pop("RIAGENDR", None)  # Ensure RIAGENDR is not renamed
label_mapping.pop("RIDAGEYR", None)  # Ensure RIAGENDR is not renamed


# Replace column names in nhanes_biomarkers with English names
nhanes_biomarkers.rename(columns=label_mapping, inplace=True)

# Ensure 'RIAGENDR' is still in the dataframe
if "RIAGENDR" not in nhanes_biomarkers.columns:
    raise KeyError("'RIAGENDR' column is missing from nhanes_biomarkers.")

# Create UI layout
app_ui = ui.page_fluid(
    ui.h2("NHANES Biomarkers Visualization"),
    # Create sidebar
    ui.layout_sidebar(
        ui.sidebar(
            # Create input selection bars
            ui.input_select(
                "biomarker_x",
                "Select Biomarker:",
                choices=list(nhanes_biomarkers.columns[1:]),
            ),
            ui.input_select(
                "heart_disease_label",
                "Select Heart Disease Label:",
                choices=list(nhanes_labels.columns[1:]),
            ),
            ui.input_select(
                "gender_filter",
                "Filter by Gender:",
                choices=["All", "Male", "Female"],
                selected="All",
            ),
            ui.input_select(
                "age_group_filter",
                "Filter by Age Group:",
                choices=[
                    "All Ages",
                    "20–30",
                    "30–40",
                    "40–50",
                    "50–60",
                    "60–70",
                    "70–80"
                ],
                selected="All Ages"
            ),

            ui.input_checkbox(
                "filter_heart_disease",
                "Filter by Heart Disease",
                value=False,
            ),
            
        ),
        # Add outputs for two separate plots
        # ui.output_plot("regression_plot"),
        ui.output_plot("kde_plot"),  
        ui.h4("Summary Statistics for Selected Biomarker"),
        ui.output_text_verbatim("summary_stats"),     
    ),
)

# Create Server logic
def server(input, output, session):
    @output
    @render.plot
    def kde_plot():
        # Get user input
        selected_biomarker_x = input.biomarker_x()
        selected_label = input.heart_disease_label()
        filter_heart_disease = input.filter_heart_disease()
        gender_filter = input.gender_filter()

        # Merge data for plotting
        try:
            merged_data = pd.merge(
                nhanes_biomarkers[["SEQN", selected_biomarker_x, "RIAGENDR", "RIDAGEYR"]],
                nhanes_labels[["id_client", selected_label]],
                left_on="SEQN",
                right_on="id_client"
            )
        except KeyError as e:
            print(f"KeyError during merge: {e}")
            raise

        # Rename columns for clarity
        merged_data.rename(
            columns={
                selected_biomarker_x: "Biomarker X",
                selected_label: "Heart Disease",
                "RIAGENDR": "Gender",
            },
            inplace=True,
        )
        # Age group filter
        age_filter = input.age_group_filter()
        # Normalize the age filter string to use regular hyphen
        age_filter = input.age_group_filter().replace("–", "-")
        age_bounds = {
            "20-30": (20, 30),
            "30-40": (30, 40),
            "40-50": (40, 50),
            "50-60": (50, 60),
            "60-70": (60, 70),
            "70-80": (70, 80)
        }
        if age_filter != "All Ages":
            low, high = age_bounds[age_filter]
            merged_data = merged_data[(merged_data["RIDAGEYR"] >= low) & (merged_data["RIDAGEYR"] < high)]
        if merged_data.empty:
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, "No data available for selected age group.", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
            return
            # Filter data if required
        if filter_heart_disease:
            merged_data = merged_data[merged_data["Heart Disease"] == 1]

        # Apply gender filter
        if gender_filter == "Male":
            merged_data = merged_data[merged_data["Gender"] == 1]
        elif gender_filter == "Female":
            merged_data = merged_data[merged_data["Gender"] == 2]

        # Plot: KDE Distribution with Gender Filter
        plt.figure(figsize=(10, 6))
        for condition, label in [(1, "Disease"), (0, "No Disease")]:
            subset = merged_data[merged_data["Heart Disease"] == condition]
            sns.kdeplot(
                subset["Biomarker X"],
                label=f"{label}",
                fill=True,
                alpha=0.5
            )
        plt.title(f"KDE of {selected_biomarker_x} Distribution by Disease Status", fontsize=14)
        plt.xlabel(selected_biomarker_x, fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(title=f"{selected_label}", fontsize=10)
        plt.grid(axis="both", linestyle="--", alpha=0.7)
        plt.tight_layout()
    @output
    @render.text
    def summary_stats():
        selected_biomarker_x = input.biomarker_x()
        selected_label = input.heart_disease_label()
        filter_heart_disease = input.filter_heart_disease()
        gender_filter = input.gender_filter()

        merged_data = pd.merge(
            nhanes_biomarkers[["SEQN", selected_biomarker_x, "RIAGENDR"]],
            nhanes_labels[["id_client", selected_label]],
            left_on="SEQN",
            right_on="id_client"
        ).rename(columns={
            selected_biomarker_x: "Biomarker X",
            selected_label: "Heart Disease",
            "RIAGENDR": "Gender"
        })

        if filter_heart_disease:
            merged_data = merged_data[merged_data["Heart Disease"] == 1]
        if gender_filter == "Male":
            merged_data = merged_data[merged_data["Gender"] == 1]
        elif gender_filter == "Female":
            merged_data = merged_data[merged_data["Gender"] == 2]

        merged_data["Heart Disease"] = merged_data["Heart Disease"].map({1: "Disease", 0: "No Disease"})

        output_str = ""

        # Grouped stats
        group_stats = {}
        for group in ["Disease", "No Disease"]:
            group_data = merged_data[merged_data["Heart Disease"] == group]["Biomarker X"].dropna()
            if group_data.empty:
                continue
            stats_dict = {
                "Count": group_data.count(),
                "Mean": group_data.mean(),
                "Median": group_data.median(),
                "Std Dev": group_data.std(),
                "Min": group_data.min(),
                "Max": group_data.max()
            }
            group_stats[group] = group_data
            output_str += f"\n--- {group} Group ---\n"
            output_str += "\n".join(f"{key}: {value:.2f}" for key, value in stats_dict.items())
            output_str += "\n"

        # Only do test if both groups exist
        if "Disease" in group_stats and "No Disease" in group_stats:
            x1 = group_stats["Disease"]
            x2 = group_stats["No Disease"]

            # Welch’s t-test
            t_stat, p_val = stats.ttest_ind(x1, x2, equal_var=False)

            # Cohen’s d
            n1, n2 = len(x1), len(x2)
            s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
            pooled_std = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
            cohen_d = (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else np.nan

            output_str += "\n--- Statistical Test ---\n"
            output_str += f"Welch's t-test p-value: {p_val:.4e}\n"
            output_str += f"Cohen's d (effect size): {cohen_d:.2f}\n"
        else:
            output_str += "\nInsufficient data to perform statistical comparison."
        if merged_data.empty:
            return "❗ No data available for the selected filters (age group, gender, disease)."
        return output_str



    # @output
    # @render.plot
    # def regression_plot():
    #     # Get user input
    #     selected_biomarker_x = input.biomarker_x()
    #     selected_biomarker_y = input.biomarker_y()
    #     selected_label = input.heart_disease_label()
    #     filter_heart_disease = input.filter_heart_disease()
    #     gender_filter = input.gender_filter()

    #     # Check for duplicate selections
    #     if selected_biomarker_x == selected_biomarker_y:
    #         raise ValueError("Please select different biomarkers for X and Y axes.")

    #     # Merge data for plotting
    #     try:
    #         merged_data = pd.merge(
    #             nhanes_biomarkers[["SEQN", selected_biomarker_x, selected_biomarker_y, "RIAGENDR"]],
    #             nhanes_labels[["id_client", selected_label]],
    #             left_on="SEQN",
    #             right_on="id_client"
    #         )
    #     except KeyError as e:
    #         print(f"KeyError during merge: {e}")
    #         raise

    #     # Rename columns for clarity
    #     merged_data.rename(
    #         columns={
    #             selected_biomarker_x: "Biomarker X",
    #             selected_biomarker_y: "Biomarker Y",
    #             selected_label: "Heart Disease",
    #             "RIAGENDR": "Gender",
    #         },
    #         inplace=True,
    #     )

    #     # Filter data if required
    #     if filter_heart_disease:
    #         merged_data = merged_data[merged_data["Heart Disease"] == 1]

    #     # Apply gender filter
    #     if gender_filter == "Male":
    #         merged_data = merged_data[merged_data["Gender"] == 1]
    #     elif gender_filter == "Female":
    #         merged_data = merged_data[merged_data["Gender"] == 2]

    #     # Plot: Regression with Gender Filter
    #     sns.lmplot(
    #         data=merged_data,
    #         x="Biomarker X",
    #         y="Biomarker Y",
    #         hue="Heart Disease",
    #         palette="Set1",
    #         ci=95,
    #         aspect=1.5,
    #         scatter_kws={"s": 10, "alpha": 0.8},
    #         legend=False
    #     )
    #     plt.title(f"{selected_biomarker_x} vs. {selected_biomarker_y}", fontsize=14)
    #     plt.xlabel(selected_biomarker_x, fontsize=12)
    #     plt.ylabel(selected_biomarker_y, fontsize=12)
    #     plt.legend(title=f"{selected_label}", fontsize=10)
    #     plt.grid(axis="both", linestyle="--", alpha=0.7)
    #     plt.tight_layout()


# Create the shiny app
app = App(app_ui, server)

