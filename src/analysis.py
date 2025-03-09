import pandas as pd

def perform_analysis(df):
    """
    Performs basic statistical analysis on the dataset.

    Args:
        df (pd.DataFrame): processed dataset.

    Returns:
        dict: Summary statistics of the dataset.
    """
    # Example analysis: basic descriptive statistics
    summary_statistics = df.describe()

    # Example: count of each sentiment category
    sentiment_counts = df['sentiment'].value_counts()  # 'sentiment' bleibt unverändert

    return {
        'summary_statistics': summary_statistics,
        'sentiment_counts': sentiment_counts
    }

def print_analysis_results(analysis_results):
    """
    Print the results of the analysis.

    Args:
        analysis_results (dict): Results from the analysis function.
    """
    print("Summary Statistics:")
    print(analysis_results['summary_statistics'])
    
    print("\nSentiment Counts:")  # 'sentiment' bleibt wie in der Rohdatei
    print(analysis_results['sentiment_counts'])