from typing import Annotated, Optional
import pandas as pd
import pingouin as pg

def perform_anova(
    data_path: Annotated[str, "Path to the input CSV file"],
    descriptor: Annotated[str, "The name of the descriptor column to analyse"],
    within_subject_factor: Annotated[str, "The name of the within-subjects factor"],
    between_subject_factor: Annotated[str, "The name of the between-subjects factor"],
    subject_id: Annotated[str, "The name of the subject identifier"],
    save_path: Annotated[Optional[str], "Path to save the ANOVA results CSV"] = None
) -> str:
    """
    Perform Mixed-design Repeated Measures ANOVA on a given descriptor.

    Returns:
        pd.DataFrame: ANOVA results as a DataFrame.
    """
    # Load data from the given file path
    data = pd.read_csv(data_path)
    
    # Ensure categorical variables are properly formatted
    data[within_subject_factor] = data[within_subject_factor].astype('category')
    data[between_subject_factor] = data[between_subject_factor].astype('category')
    data[subject_id] = data[subject_id].astype('category')

    # Perform Mixed-design Repeated Measures ANOVA
    anova_results = pg.mixed_anova(dv=descriptor,
                                   within=within_subject_factor,
                                   between=between_subject_factor,
                                   subject=subject_id,
                                   data=data,
                                   correction='auto')  # Apply Greenhouse-Geisser correction if needed
    print(anova_results)

    # Save results to CSV if a save path is provided
    if save_path:
        anova_results.to_csv(save_path, index=False)
        return f"The Mixed-design Repeated Measures ANOVA results on {descriptor} have been saved to {save_path}."
    else: 
        return anova_results.to_dict(orient='records')


def perform_tukey_test(
    data_path: Annotated[str, "Path to the input CSV file"],
    descriptor: Annotated[str, "The name of the descriptor column to analyse"],
    between_subject_factor: Annotated[str, "The name of the between-subjects factor"],
    subject_id: Annotated[str, "The name of the subject identifier"],
    save_path: Annotated[Optional[str], "Path to save the Tukey-Kramer results CSV"] = None
) -> str:
    """
    Perform Post-hoc Tukey-Kramer test on a given descriptor.

    Returns:
        pd.DataFrame: Tukey-Kramer test results as a DataFrame.
    """
    # Load data from the given file path
    data = pd.read_csv(data_path)
    
    # Aggregate data by averaging over time for each plant
    aggregated_data = data.groupby([subject_id, between_subject_factor], observed=True)[descriptor].mean().reset_index()

    # Perform Tukey-Kramer post-hoc test
    tukey_results = pg.pairwise_tukey(data=aggregated_data,
                                      dv=descriptor,
                                      between=between_subject_factor)
    print(tukey_results)

    # Save results to CSV if a save path is provided
    if save_path:
        tukey_results.to_csv(save_path, index=False)
        return f"The Post-hoc Tukey-Kramer test results on {descriptor} have been saved to {save_path}."
    else: 
        return tukey_results.to_dict(orient='records')
