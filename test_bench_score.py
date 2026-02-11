from viescore import VIEScore
import PIL
import os
import megfile
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import sys
import csv
import threading
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import itertools
import json
GROUPS = [
    "t2i_long", "t2i_complex", "edit_multi", "edit_complex"  
]

    
def process_single_item_edit(group_name, item, vie_score, max_retries=10):

    instruction = item['instruction']
    key = item['key']
    
    src_image_path = f"{save_path}/fullset/{group_name}/{key}_SRCIMG.png"
    save_path_item = f"{save_path}/fullset/{group_name}/{key}.png"

    #for retry in range(max_retries):
        #try:
    pil_image_raw =Image.open(megfile.smart_open(src_image_path, 'rb'))
    pil_image_edited = Image.open(megfile.smart_open(save_path_item, 'rb')).convert("RGB").resize((pil_image_raw.size[0], pil_image_raw.size[1]))
    text_prompt = instruction
    score_list = vie_score.evaluate([pil_image_raw, pil_image_edited], text_prompt)
    sementics_score, quality_score, overall_score = score_list
    print(f"sementics_score: {sementics_score}, quality_score: {quality_score}, overall_score: {overall_score}, instruction: {instruction}")
    
    return {"data_type":group_name, "source_image": src_image_path, "generated_image": save_path_item, "instruction": instruction, "sementics_score": sementics_score, "quality_score": quality_score}

        # except Exception as e:
        #     if retry < max_retries - 1:
        #         wait_time = (retry + 1) * 2
        #         print(f"Error processing {save_path_item} (attempt {retry + 1}/{max_retries}): {e}")
        #         print(f"Waiting {wait_time} seconds before retry...")
        #         time.sleep(wait_time)
        #     else:
        #         print(f"Failed to process {save_path_item} after {max_retries} attempts: {e}")
        #         return

def process_single_item_t2i(group_name, item, vie_score, max_retries=10):

    instruction = item['instruction']
    key = item['key']
    
    save_path_item = f"{save_path}/fullset/{group_name}/{key}.png"

    for retry in range(max_retries):
        try:
            pil_image_generated = Image.open(megfile.smart_open(save_path_item, 'rb')).convert("RGB")
            text_prompt = instruction
            score_list = vie_score.evaluate([pil_image_generated], text_prompt)
            sementics_score, quality_score, overall_score = score_list
            print(f"sementics_score: {sementics_score}, quality_score: {quality_score}, overall_score: {overall_score}, instruction: {instruction}")
            
            return {"data_type": group_name, "source_image": None, "generated_image": save_path_item, "instruction": instruction, "sementics_score": sementics_score, "quality_score": quality_score}

        except Exception as e:
            if retry < max_retries - 1:
                wait_time = (retry + 1) * 2
                print(f"Error processing {save_path_item} (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to process {save_path_item} after {max_retries} attempts: {e}")
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="benchmark_results/")
    parser.add_argument("--backbone", type=str, default="qwen25vl")
    args = parser.parse_args()
    model_name = os.path.basename(args.image_path)
    save_path_dir = os.path.dirname(args.image_path)
    evaluate_group = [model_name]
    backbone = args.backbone

    vie_score_edit = VIEScore(backbone=backbone, task="tie")
    vie_score_t2i = VIEScore(backbone=backbone, task="t2i")
    max_workers = 5

    for model_name in evaluate_group:
        save_path = os.path.join(save_path_dir, model_name)
        save_path_new = os.path.join(save_path_dir, model_name, backbone, "eval_results_AWQ")
        all_csv_list = []  # Store all results for final combined CSV
        
        # Load existing processed samples from final CSV if it exists
        processed_samples = set()
        final_csv_path = os.path.join(save_path_new, f"{model_name}_combined_score.csv")
        if megfile.smart_exists(final_csv_path):
            with megfile.smart_open(final_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Create a unique identifier for each sample
                    sample_key = (row['source_image'], row['edited_image'])
                    processed_samples.add(sample_key)
            print(f"Loaded {len(processed_samples)} processed samples from existing CSV")

        for group_name in GROUPS:
            group_csv_list = []
            
            group_csv_path = os.path.join(save_path_new, f"{model_name}_{group_name}_score.csv")
            if megfile.smart_exists(group_csv_path):
                with megfile.smart_open(group_csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    group_results = list(reader)
                    group_csv_list.extend(group_results)
            
                print(f"Loaded existing results for {model_name} - {group_name}")
            
            print(f"Processing group: {group_name}")
            print(f"Processing model: {model_name}")
            
            
            json_file = "benchmark/semgen_bench.jsonl"
            with open(json_file, 'r', encoding='utf-8') as file:
                for key, meta_data in enumerate(file):
                    item = json.loads(meta_data.strip())
                    item['key'] = key
                    item['instruction'] = item[f'{group_name}']
                    
                    if "t2i" in group_name:
                        save_path_fullset_result_image = f"{save_path}/fullset/{group_name}/{key}.png"

                        if not megfile.smart_exists(save_path_fullset_result_image):
                            print(f"Skipping {key}: Source or edited image does not exist")
                            print(f"{save_path_fullset_result_image} does not exist")
                            continue

                        # Check if this sample has already been processed
                        sample_key = save_path_fullset_result_image
                        exists = sample_key in processed_samples
                        if exists:
                            print(f"Skipping already processed sample: {key}")
                            continue

                        result = process_single_item_t2i(group_name, item, vie_score_edit)

                    else:
                        save_path_fullset_source_image = f"{save_path}/fullset/{group_name}/{key}_SRCIMG.png"
                        save_path_fullset_result_image = f"{save_path}/fullset/{group_name}/{key}.png"

                        if not megfile.smart_exists(save_path_fullset_result_image) or not megfile.smart_exists(save_path_fullset_source_image):
                            print(f"Skipping {key}: Source or edited image does not exist")
                            print(f"{save_path_fullset_source_image} or {save_path_fullset_result_image} does not exist")
                            continue

                        # Check if this sample has already been processed
                        sample_key = (save_path_fullset_source_image, save_path_fullset_result_image)
                        exists = sample_key in processed_samples
                        if exists:
                            print(f"Skipping already processed sample: {key}")
                            continue

                        result = process_single_item_edit(group_name, item, vie_score_edit)

                    if result:
                        group_csv_list.append(result)

            # Save group-specific CSV
            group_csv_path = os.path.join(save_path_new, f"{model_name}_{group_name}_score.csv")
            with megfile.smart_open(group_csv_path, 'w', newline='') as f:
                fieldnames = ["data_type", "source_image", "generated_image", "instruction", "sementics_score", "quality_score"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in group_csv_list:
                    writer.writerow(row)
            all_csv_list.extend(group_csv_list)

            print(f"Saved group CSV for {group_name}, length： {len(group_csv_list)}")

        # After processing all groups, calculate and save combined results
        if not all_csv_list:
            print(f"Warning: No results for model {model_name}, skipping combined CSV generation")
            continue

        # Save combined CSV
        combined_csv_path = os.path.join(save_path_new, f"{model_name}_combined_score.csv")
        with megfile.smart_open(combined_csv_path, 'w', newline='') as f:
            fieldnames = ["data_type", "source_image", "generated_image", "instruction", "sementics_score", "quality_score"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_csv_list:
                writer.writerow(row)

                
            
            
