# EVAL

Our evaluation process consists of the following steps:
1. Prepare the Environment
   - Install required dependencies:
     ```bash
     conda env create -f qwen_vl_environment.yml
     conda activate qwen_vl
     ```

2. Generate and Organize Your Images
   - Generate images following the example code in `generate_bench_img.py`
   - Organize your generated images in the following directory structure:
     ```
     results/
     ├── method_name/
     │   └── fullset/
     │       └── edit_task/
     │           ├──key1.png
     │           ├── key2.png
     │           └── ...
     ```

3. Evaluate using Qwen2.5VL-72B-Instruct-AWQ
     ```bash
     python test_bench_score.py --model_name your_method --save_path --backbone qwen25vl
     ```

4. Analyze your results and obtain scores across all dimensions
   - Run the analysis script to get scores for semantics, quality, and overall performance:
     ```bash
     python calculate_statistics.py --model_name your_method --save_path /path/to/results --backbone qwen25vl
     ```
   - This will output scores broken down by edit category and provide aggregate metrics

# Acknowledgements

This project builds upon and adapts code from the following excellent repositories:

- [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit): A Practical Framework for General Image Editing

- [VIEScore](https://github.com/TIGER-AI-Lab/VIEScore): A visual instruction-guided explainable metric for evaluating conditional image synthesis

We thank the authors of these repositories for making their code publicly available.

