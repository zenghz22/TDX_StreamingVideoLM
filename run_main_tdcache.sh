run_single_encode_multi_decode(){
    local chunk_size=$1
    local encode_memory=$2
    local encode_window=$3
    local decode_select=$4
    local decode_memory=$5
    local decode_policy=$6
    local name="SEMD_C${chunk_size}_M${encode_memory}_W${encode_window}_S${decode_select}_DM${decode_memory}_DP${decode_policy}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py \
    --mode "encode_decode" --chunk_size $chunk_size --encode_memory $encode_memory --encode_window $encode_window --decode_select $decode_select \
    --plot_file "../results/plots/${name}.png"   \
    --encrypt \
    --decode_memory $decode_memory --decode_policy $decode_policy \
    > ../results/logs/${name}.log 2>&1
}

run_single_encode_multi_decode 1 1024 64 32 2048 0
run_single_encode_multi_decode 1 1024 64 32 2048 1
run_single_encode_multi_decode 1 1024 64 32 2048 2
run_single_encode_multi_decode 1 1024 64 32 2048 3
run_single_encode_multi_decode 1 1024 64 32 2048 4
run_single_encode_multi_decode 1 1024 64 32 2048 5