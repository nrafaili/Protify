#!/usr/bin/env python
"""
Utility script to list Protify supported models and datasets.
"""

import argparse
import sys

def list_models(show_standard_only=False):
    """List available models with descriptions if available"""
    try:
        from .base_models.get_base_models import currently_supported_models, standard_models
        from .base_models.model_descriptions import model_descriptions
        
        if show_standard_only:
            models_to_show = standard_models
            print("\n=== Standard Models ===\n")
        else:
            models_to_show = currently_supported_models
            print("\n=== All Supported Models ===\n")
        
        # Calculate maximum widths for formatting
        max_name_len = max(len(name) for name in models_to_show)
        max_type_len = max(len(model_descriptions.get(name, {}).get('type', 'Unknown')) for name in models_to_show if name in model_descriptions)
        max_size_len = max(len(model_descriptions.get(name, {}).get('size', 'Unknown')) for name in models_to_show if name in model_descriptions)
        
        # Print header
        print(f"{'Model':<{max_name_len+2}}{'Type':<{max_type_len+2}}{'Size':<{max_size_len+2}}Description")
        print("-" * (max_name_len + max_type_len + max_size_len + 50))
        
        # Print model information
        for model_name in models_to_show:
            if model_name in model_descriptions:
                model_info = model_descriptions[model_name]
                print(f"{model_name:<{max_name_len+2}}{model_info.get('type', 'Unknown'):<{max_type_len+2}}{model_info.get('size', 'Unknown'):<{max_size_len+2}}{model_info.get('description', 'No description available')}")
            else:
                print(f"{model_name:<{max_name_len+2}}{'Unknown':<{max_type_len+2}}{'Unknown':<{max_size_len+2}}No description available")
    
    except ImportError as e:
        print(f"Error loading model information: {e}")
        print("\n=== Models ===\n")
        try:
            from .base_models.get_base_models import currently_supported_models, standard_models
            
            if show_standard_only:
                for model_name in standard_models:
                    print(f"- {model_name}")
            else:
                for model_name in currently_supported_models:
                    print(f"- {model_name}")
        except ImportError:
            print("Could not load model lists. Please check your installation.")

def list_datasets(show_standard_only=False):
    """List available datasets with descriptions if available"""
    try:
        from .data.supported_datasets import supported_datasets, standard_data_benchmark
        from .data.dataset_descriptions import dataset_descriptions
        
        if show_standard_only:
            datasets_to_show = {name: supported_datasets[name] for name in standard_data_benchmark if name in supported_datasets}
            print("\n=== Standard Benchmark Datasets ===\n")
        else:
            datasets_to_show = supported_datasets
            print("\n=== All Supported Datasets ===\n")
        
        # Calculate maximum widths for formatting
        max_name_len = max(len(name) for name in datasets_to_show)
        max_type_len = max(len(dataset_descriptions.get(name, {}).get('type', 'Unknown')) for name in datasets_to_show if name in dataset_descriptions)
        max_task_len = max(len(dataset_descriptions.get(name, {}).get('task', 'Unknown')) for name in datasets_to_show if name in dataset_descriptions)
        
        # Print header
        print(f"{'Dataset':<{max_name_len+2}}{'Type':<{max_type_len+2}}{'Task':<{max_task_len+2}}Description")
        print("-" * (max_name_len + max_type_len + max_task_len + 50))
        
        # Print dataset information
        for dataset_name in datasets_to_show:
            if dataset_name in dataset_descriptions:
                dataset_info = dataset_descriptions[dataset_name]
                print(f"{dataset_name:<{max_name_len+2}}{dataset_info.get('type', 'Unknown'):<{max_type_len+2}}{dataset_info.get('task', 'Unknown'):<{max_task_len+2}}{dataset_info.get('description', 'No description available')}")
            else:
                print(f"{dataset_name:<{max_name_len+2}}{'Unknown':<{max_type_len+2}}{'Unknown':<{max_task_len+2}}No description available")
    
    except ImportError as e:
        print(f"Error loading dataset information: {e}")
        print("\n=== Datasets ===\n")
        try:
            from .data.supported_datasets import supported_datasets, standard_data_benchmark
            
            if show_standard_only:
                for dataset_name in standard_data_benchmark:
                    if dataset_name in supported_datasets:
                        print(f"- {dataset_name}: {supported_datasets[dataset_name]}")
            else:
                for dataset_name, dataset_source in supported_datasets.items():
                    print(f"- {dataset_name}: {dataset_source}")
        except ImportError:
            print("Could not load dataset lists. Please check your installation.")

def main():
    """Main function to run the script from command line"""
    parser = argparse.ArgumentParser(description='List Protify supported models and datasets')
    parser.add_argument('--models', action='store_true', help='List supported models')
    parser.add_argument('--datasets', action='store_true', help='List supported datasets')
    parser.add_argument('--standard-only', action='store_true', help='Show only standard models/datasets')
    parser.add_argument('--all', action='store_true', help='Show both models and datasets')
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1 or args.all:
        list_models(args.standard_only)
        print("\n" + "="*80 + "\n")
        list_datasets(args.standard_only)
        return
    
    if args.models:
        list_models(args.standard_only)
    
    if args.datasets:
        list_datasets(args.standard_only)

if __name__ == "__main__":
    main() 