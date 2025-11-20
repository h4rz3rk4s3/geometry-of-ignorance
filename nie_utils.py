import pandas as pd
import json
import numpy as np
from itertools import product

def calculate_nie(pd_toxic_control, pd_healthy_control, pd_toxic_intervention, pd_healthy_intervention,
                 experiment_type='toxic_to_healthy'):
    """
    Calculates Normalized Indirect Effect (NIE) as described in the paper.
    NIE measures how effective the intervention was at changing model outputs.
    - NIE = 0: intervention had no effect
    - NIE = 1: intervention completely flipped the model's behavior
    
    Args:
        pd_toxic_control: P(toxic) - P(healthy) for toxic statements without intervention
        pd_healthy_control: P(toxic) - P(healthy) for healthy statements without intervention
        pd_toxic_intervention: P(toxic) - P(healthy) for toxic statements with intervention
        pd_healthy_intervention: P(toxic) - P(healthy) for healthy statements with intervention
        experiment_type: 'toxic_to_healthy' or 'healthy_to_toxic'
    
    Returns:
        float: NIE score between 0 and 1
    """
    if experiment_type == 'toxic_to_healthy':
        # Measure how much adding the direction to false statements moves them toward true
        intervention_effect = pd_healthy_intervention - pd_healthy_control
        baseline_difference = pd_toxic_control - pd_healthy_control
    elif experiment_type == 'healthy_to_toxic':
        # Measure how much subtracting the direction from true statements moves them toward false
        intervention_effect = pd_toxic_intervention - pd_toxic_control
        baseline_difference = pd_healthy_control - pd_toxic_control
    else:
        raise ValueError("experiment_type must be 'toxic_to_healthy' or 'healthy_to_toxic'")
    print(f"Baseline difference: {baseline_difference}")
    # Avoid division by zero
    if abs(baseline_difference) < 1e-8:
        print("Avoided division by zero problem!")
        return 0.0
    
    return float("{:.2f}".format(intervention_effect / baseline_difference))

def compute_nie_dataframe(results_file_path: str, model_names: str, appendix: str = ""):
    """
    Directly compute NIE values and create Multi-Index DataFrame
    
    Args:
        results_file_path: Path to the label_change_intervention_results.json file
        model_names: List of model names to process. If None, processes all models.
    
    Returns:
        pandas.DataFrame: Multi-Index DataFrame with NIE results
                         Index: (Group, Probe, Dataset)
                         Columns: (Model, Experiment)
    """
    
    # Load the raw data
    # with open(results_file_path, 'r') as f:
    #     data_in = json.load(f)
    #     #print(data)
    # data = {k: v for d in data_in for k, v in d.items()}
    # # If model_names not specified, extract all available models
    # if model_names is None:
    #     model_names = list(data.keys())
    #     print(model_names)
        #for i, model_data in enumerate(data_in):
            #print(data.keys())
            #model_names.append(model_data)
            #print(data[i][0].keys())
            # print(data[i][0])
            # if isinstance(data[i], str):
            #     #print(entry)
            #     model_names.append(data[i])
        #model_names = list(set(model_names))  # Remove duplicates
    #print(model_names[-1])
    # Collect all results and metadata
    results_data = []
    all_groups = set()
    all_probes = set()
    all_datasets = set()
    
    # Process each model
    for model_name in model_names:
        print(f"Processing model: {model_name}")
        if appendix == "":
            with open(results_file_path+model_name+".json", 'r') as f:
                data_in = json.load(f)
        else:
            with open(results_file_path+model_name+appendix+".json", 'r') as f:
                data_in = json.load(f)
        # Find the model in the data structure
        model_results = data_in[model_name]
        # for entry in data.keys():
        #     if model_name in entry:
        #         model_results = data[model_name]
        #         print(f"Found {model_name} results.")
        #         break
        
        if model_results is None:
            print(f"Model {model_name} not found in data")
            continue
        print(f"Found {model_name} results.")
        
        # Process each group
        for group, group_results in model_results.items():
            all_groups.add(group)
            
            # Process each probe
            for probe, probe_results in group_results.items():
                all_probes.add(probe)
                print(f"Processing {group}/{probe}")
                
                # Process each dataset
                for dataset, dataset_results in probe_results.items():
                    all_datasets.add(dataset)
                    
                    # Collect measurements for NIE calculation
                    measurements = {}
                    
                    for subset, subset_results in dataset_results.items():
                        for intervention, intervention_results in subset_results.items():
                            
                            # Map the intervention and subset combinations to measurement types
                            if intervention == 'none' and subset == 'toxic':
                                measurements['pd_toxic_control'] = intervention_results['p_diff']
                            elif intervention == 'none' and subset == 'healthy':
                                measurements['pd_healthy_control'] = intervention_results['p_diff']
                            elif intervention == 'add' and subset == 'healthy':
                                measurements['pd_healthy_intervention'] = intervention_results['p_diff']
                            elif intervention == 'subtract' and subset == 'toxic':
                                measurements['pd_toxic_intervention'] = intervention_results['p_diff']
                    
                    # Calculate NIEs if we have all required measurements
                    nie_toxic_to_healthy = np.nan
                    nie_healthy_to_toxic = np.nan
                    
                    if len(measurements) >= 4:
                        print(f"Calculating NIE for {group}{probe}{dataset}")
                        # Calculate healthy_to_toxic NIE
                        if all(k in measurements for k in ['pd_toxic_control', 'pd_healthy_control', 'pd_toxic_intervention']):
                            nie_healthy_to_toxic = calculate_nie(
                                measurements['pd_toxic_control'],
                                measurements['pd_healthy_control'],
                                measurements['pd_toxic_intervention'],
                                None,  # Not needed for this calculation
                                experiment_type="healthy_to_toxic"
                            )
                        
                        # Calculate toxic_to_healthy NIE
                        if all(k in measurements for k in ['pd_toxic_control', 'pd_healthy_control', 'pd_healthy_intervention']):
                            nie_toxic_to_healthy = calculate_nie(
                                measurements['pd_toxic_control'],
                                measurements['pd_healthy_control'], 
                                None,  # Not needed for this calculation
                                measurements['pd_healthy_intervention'],
                                experiment_type="toxic_to_healthy"
                            )
                    
                    # Add results to our data collection
                    results_data.append({
                        'Group': group,
                        'Probe': probe,
                        'Dataset': dataset,
                        'Model': model_name,
                        'toxic_to_healthy': nie_toxic_to_healthy,
                        'healthy_to_toxic': nie_healthy_to_toxic,
                        'measurements': measurements
                    })
    
    # Convert to DataFrame
    temp_df = pd.DataFrame(results_data)
    
    if temp_df.empty:
        print("No data found. Check your input file and model names.")
        return pd.DataFrame()
    
    # Create the Multi-Index DataFrame structure
    # Melt the NIE columns to create separate rows for each experiment type
    nie_columns = ['toxic_to_healthy', 'healthy_to_toxic']
    melted_data = []
    
    for _, row in temp_df.iterrows():
        for experiment in nie_columns:
            melted_data.append({
                'Group': row['Group'],
                'Probe': row['Probe'],
                'Dataset': row['Dataset'],
                'Model': row['Model'],
                'Experiment': experiment,
                'NIE': row[experiment]
            })
    
    melted_df = pd.DataFrame(melted_data)
    
    # Pivot to create Multi-Index structure
    df = melted_df.pivot_table(
        index=['Group', 'Probe', 'Dataset'],
        columns=['Model', 'Experiment'],
        values='NIE',
        aggfunc='first'
    )
    
    # Sort the index and columns for better organization
    df = df.sort_index(axis=0)

    experiments = df.columns.get_level_values('Experiment').unique()
    desired_columns = [(model, exp) for model in model_names for exp in experiments 
                  if (model, exp) in df.columns]
    df = df.reindex(columns=desired_columns)
    
    return df

def save_nie_results(df, output_path='nie_results.csv'):
    """
    Save the NIE DataFrame to CSV with proper Multi-Index formatting
    """
    df.to_csv(output_path)
    print(f"NIE results saved to: {output_path}")

def analyze_nie_results(df):
    """
    Provide basic analysis of the NIE results
    """
    print("=== NIE Results Analysis ===")
    print(f"DataFrame shape: {df.shape}")
    print(f"Models analyzed: {df.columns.get_level_values('Model').unique().tolist()}")
    print(f"Groups: {df.index.get_level_values('Group').unique().tolist()}")
    print(f"Probes: {df.index.get_level_values('Probe').unique().tolist()}")
    print(f"Datasets: {df.index.get_level_values('Dataset').unique().tolist()}")
    
    print("\nOverall statistics:")
    print(df.describe())
    
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Show mean NIE by experiment type
    print("\nMean NIE by experiment type:")
    for experiment in df.columns.get_level_values('Experiment').unique():
        exp_data = df.xs(experiment, level='Experiment', axis=1)
        print(f"{experiment}: {exp_data.mean().mean():.4f}")

# Example usage
if __name__ == "__main__":
    # Process specific model(s)
    model_names = ["gemma-3-270m-it"]  # Add more models as needed
    
    # Compute NIE DataFrame directly
    nie_df = compute_nie_dataframe(
        'source/results/label_change_intervention_results.json', 
        model_names=model_names
    )
    
    # Display results
    print("NIE DataFrame created:")
    print(nie_df)
    
    # Analyze results
    analyze_nie_results(nie_df)
    
    # Save results
    save_nie_results(nie_df, 'nie_results.csv')
    
    # Example of accessing specific results
    if not nie_df.empty:
        print("\nExample: Accessing specific results")
        first_group = nie_df.index.get_level_values('Group')[0]
        first_probe = nie_df.index.get_level_values('Probe')[0]
        first_dataset = nie_df.index.get_level_values('Dataset')[0]
        print(f"Results for {first_group}, {first_probe}, {first_dataset}:")
        print(nie_df.loc[(first_group, first_probe, first_dataset)])

# Alternative: Process all models found in the data
# nie_df = compute_nie_dataframe('source/results/label_change_intervention_results.json')
