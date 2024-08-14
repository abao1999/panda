# Set variable to current directory
current_dir=$(dirname "$0")

# # In-domain evaluation
# # Results will be saved in: $current_dir/evaluation/results/chronos_dysts-t5-small-in-domain.csv
# python scripts/evaluation/evaluate.py \
#     $current_dir/evaluation/configs/dysts_in-domain.yaml \
#     $current_dir/evaluation/results/chronos_dysts-t5-small-in-domain.csv \
#     --chronos-model-id "amazon/chronos-t5-small" \
#     --batch-size=32 \
#     --device=cuda:0 \
#     --num-samples 20

# # "Zero-shot" evaluation
# # Results will be saved in: $current_dir/evaluation/results/chronos_dysts-t5-small-zero-shot.csv
# python scripts/evaluation/evaluate.py \
#     $current_dir/evaluation/configs/dysts_holdout.yaml \
#     $current_dir/evaluation/results/chronos_dysts-t5-small-zero-shot.csv \
#     --chronos-model-id "amazon/chronos-t5-small" \
#     --batch-size=32 \
#     --device=cuda:0 \
#     --num-samples 20

# # ================= CHRONOS BENCHMARKS =================
# In-domain evaluation
python scripts/evaluation/chronos_benchmarks/evaluate.py \
    $current_dir/evaluation/chronos_benchmarks/configs/in-domain.yaml \
    $current_dir/evaluation/chronos_benchmarks/results/chronos-t5-small-in-domain.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20

# # Zero-shot evaluation
# # Results will be saved in: $current_dir/evaluation/results/chronos-t5-small-zero-shot.csv
# python scripts/evaluation/chronos_benchmarks/evaluate.py \
#     $current_dir/evaluation/chronos_benchmarks/configs/zero-shot.yaml \
#     $current_dir/evaluation/chronos_benchmarks/results/chronos-t5-small-zero-shot.csv \
#     --chronos-model-id "amazon/chronos-t5-small" \
#     --batch-size=32 \
#     --device=cuda:0 \
#     --num-samples 20