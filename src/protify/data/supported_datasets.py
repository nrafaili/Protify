
supported_datasets = {
    'EC': 'GleghornLab/EC_reg',
    'GO-CC': 'GleghornLab/CC_reg',
    'GO-BP': 'GleghornLab/BP_reg',
    'GO-MF': 'GleghornLab/MF_reg',
    'MB': 'GleghornLab/MB_reg',
    'DeepLoc-2': 'GleghornLab/DL2_reg',
    'DeepLoc-10': 'GleghornLab/DL10_reg',
    'enzyme-kcat': 'GleghornLab/enzyme_kcat',
    'solubility': 'GleghornLab/solubility_prediction',
    'localization': 'GleghornLab/localization_prediction',
    'temperature-stability': 'GleghornLab/temperature_stability',
    'peptide-HLA-MHC-affinity': 'GleghornLab/peptide_HLA_MHC_affinity_ppi',
    'optimal-temperature': 'GleghornLab/optimal_temperature',
    'optimal-ph': 'GleghornLab/optimal_ph',
    'material-production': 'GleghornLab/material_production',
    'fitness-prediction': 'GleghornLab/fitness_prediction',
    'number-of-folds': 'GleghornLab/fold_prediction',
    'cloning-clf': 'GleghornLab/cloning_clf',
    'stability-prediction': 'GleghornLab/stability_prediction',
    'human-ppi': 'GleghornLab/HPPI',
    'SecondaryStructure-3': 'GleghornLab/SS3',
    'SecondaryStructure-8': 'GleghornLab/SS8',
    'fluorescence-prediction': 'GleghornLab/fluorescence_prediction',
    'plastic': 'GleghornLab/plastic_degradation_benchmark',
    'gold-ppi': 'Synthyra/bernett_gold_ppi',
    'synthyra-ppi': 'Synthyra/ppi_set_v5',
}

internal_synthyra_datasets = {
    'synthyra-ppi': 'Synthyra/ppi_set_v5',
}

possible_with_vector_reps = [
    'EC',
    'GO-CC',
    'GO-BP',
    'GO-MF',
    'MB',
    'DeepLoc-2',
    'DeepLoc-10',
    'enzyme-kcat',
    'solubility',
    'localization',
    'temperature-stability',
    'peptide-HLA-MHC-affinity',
    'optimal-temperature',
    'optimal-ph',
    'material-production',
    'fitness-prediction',
    'number-of-folds',
    'cloning-clf',
    'stability-prediction',
    'human-ppi',
    'gold-ppi',
]

standard_data_benchmark = [
    'EC',
    'GO-CC',
    'GO-BP',
    'GO-MF',
    'MB',
    'DeepLoc-2',
    'DeepLoc-10',
    'enzyme-kcat',
    'optimal-temperature',
    'optimal-ph',
    'fitness-prediction',
]

testing = [
    'EC', # multilabel
    'DeepLoc-2', # 
    'DeepLoc-10', # multiclass
    'enzyme-kcat', # regression
    'human-ppi', # ppi
]


residue_wise_problems = [
    'SecondaryStructure-3',
    'SecondaryStructure-8',
    'fluorescence-prediction',
]