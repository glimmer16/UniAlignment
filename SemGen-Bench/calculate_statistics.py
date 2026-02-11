import megfile
import os
import pandas as pd
from collections import defaultdict
import sys
import numpy as np
import math

GROUPS = [
    "t2i_long", "t2i_complex", "edit_multi", "edit_complex"
]

def analyze_scores(save_path_dir, evaluate_group):

    save_path_new = save_path_dir

    group_scores_semantics = defaultdict(lambda: defaultdict(list))
    group_scores_quality = defaultdict(lambda: defaultdict(list))
    group_scores_overall = defaultdict(lambda: defaultdict(list))
    
    for group_name in GROUPS:

        csv_path = os.path.join(save_path_new, f"{evaluate_group[0]}_{group_name}_score.csv")
        csv_file = megfile.smart_open(csv_path)
        df = pd.read_csv(csv_file)
        
        filtered_semantics_scores = []
        filtered_quality_scores = []
        filtered_overall_scores = []
        
        for _, row in df.iterrows():
            
            semantics_score = row['sementics_score']
            quality_score = row['quality_score']
            
            overall_score = math.sqrt(semantics_score * quality_score)
            
            filtered_semantics_scores.append(semantics_score)
            filtered_quality_scores.append(quality_score)
            filtered_overall_scores.append(overall_score)
        
        avg_semantics_score = np.mean(filtered_semantics_scores)
        avg_quality_score = np.mean(filtered_quality_scores)
        avg_overall_score = np.mean(filtered_overall_scores)
        group_scores_semantics[evaluate_group[0]][group_name] = avg_semantics_score
        group_scores_quality[evaluate_group[0]][group_name] = avg_quality_score
        group_scores_overall[evaluate_group[0]][group_name] = avg_overall_score


    print("\n--- Overall Model Averages ---")

    print("\nSemantics:")
    for model_name in evaluate_group:
        model_scores = [group_scores_semantics[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_semantics[model_name]["avg_semantics"] = model_avg
    
    print("\nQuality:")
    for model_name in evaluate_group:
        model_scores = [group_scores_quality[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_quality[model_name]["avg_quality"] = model_avg

    print("\nOverall:")
    for model_name in evaluate_group:
        model_scores = [group_scores_overall[model_name][group] for group in GROUPS]
        model_avg = np.mean(model_scores)
        group_scores_overall[model_name]["avg_overall"] = model_avg

    
    return group_scores_semantics, group_scores_quality, group_scores_overall

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="model_name")
    parser.add_argument("--save_path", type=str, default="benchmark_results")
    parser.add_argument("--backbone", type=str, default="qwen25vl")
    args = parser.parse_args()
    model_name = args.model_name
    save_path_dir = args.save_path
    evaluate_group = [args.model_name]
    backbone = args.backbone

    save_path_new = os.path.join(save_path_dir, model_name, backbone, "eval_results_AWQ")

    print("\nOverall:")
   
    for model_name in evaluate_group:
        group_scores_semantics, group_scores_quality, group_scores_overall = analyze_scores(save_path_new, [model_name])
    for group_name in GROUPS:
        print(f"{group_name}: {group_scores_semantics[model_name][group_name]:.3f}, {group_scores_quality[model_name][group_name]:.3f}, {group_scores_overall[model_name][group_name]:.3f}")

    print(f"Average: {group_scores_semantics[model_name]['avg_semantics']:.3f}, {group_scores_quality[model_name]['avg_quality']:.3f}, {group_scores_overall[model_name]['avg_overall']:.3f}")
