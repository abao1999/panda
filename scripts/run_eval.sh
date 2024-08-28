# Set variable to current directory
current_dir=$(dirname "$0")

# In-domain evaluation
# python scripts/evaluate.py \
#         eval.model_id=/stor/work/AMDG_Gilpin_Summer2024/checkpoints/checkpoint-80000 \
#         eval.data_dir=/stor/work/AMDG_Gilpin_Summer2024/data \

# python scripts/evaluate.py \
#         eval.model_id=amazon/chronos-t5-large \
#         eval.data_dir=/stor/work/AMDG_Gilpin_Summer2024/data \

# python scripts/evaluate.py \
#         eval.model_id=/stor/work/AMDG_Gilpin_Summer2024/checkpoints/checkpoint-40000 \
#         eval.data_dir=/stor/work/AMDG_Gilpin_Summer2024/data \

# python scripts/evaluate.py \
#         eval.model_id=/stor/work/AMDG_Gilpin_Summer2024/checkpoints/checkpoint-80000 \
#         eval.data_dir=/stor/work/AMDG_Gilpin_Summer2024/data \
#         eval.split=test \

python scripts/evaluate.py \
        eval.model_id=amazon/chronos-t5-large \
        eval.data_dir=/stor/work/AMDG_Gilpin_Summer2024/data \
        eval.split=test \