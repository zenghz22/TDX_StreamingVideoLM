run_encode(){
    local chunk_size=$1
    local encode_memory=$2
    local encode_window=$3
    local name="C${chunk_size}_encode_M${encode_memory}_W${encode_window}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --mode "encode" --chunk_size $chunk_size --encode_memory $encode_memory --encode_window $encode_window --plot_file "../results/plots/${name}.png" > ../results/logs/${name}.log 2>&1
}

run_decode(){
    local chunk_size=$1
    local decode_select=$2
    local encode_window=$3
    local name="C${chunk_size}_decode_S${decode_select}_W${encode_window}"
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --mode "decode" --chunk_size $chunk_size --decode_select $decode_select --encode_window $encode_window --plot_file "../results/plots/${name}.png"   > ../results/logs/${name}.log 2>&1
}

run_encode_decode(){
    local chunk_size=$1
    local encode_memory=$2
    local encode_window=$3
    local decode_select=$4
    local name="C${chunk_size}_encode_decode_M${encode_memory}_W${encode_window}_S${decode_select}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --mode "encode_decode" --chunk_size $chunk_size --encode_memory $encode_memory --encode_window $encode_window --decode_select $decode_select --plot_file "../results/plots/${name}.png"   > ../results/logs/${name}.log 2>&1
}


run_encode_encrypt(){
    local chunk_size=$1
    local encode_memory=$2
    local encode_window=$3
    local name="C${chunk_size}_encode_encrypt_M${encode_memory}_W${encode_window}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py \
    --mode "encode" --chunk_size $chunk_size --encode_memory $encode_memory --encode_window $encode_window \
    --plot_file "../results/plots/${name}.png"   \
    --encrypt \
    > ../results/logs/${name}.log 2>&1
}

run_decode_decrypt(){
    local chunk_size=$1
    local decode_select=$2
    local encode_window=$3
    local name="C${chunk_size}_decode_decrypt_S${decode_select}_W${encode_window}"
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py \
    --mode "decode" --chunk_size $chunk_size --decode_select $decode_select --encode_window $encode_window \
    --plot_file "../results/plots/${name}.png"   \
    --encrypt \
    > ../results/logs/${name}.log 2>&1
}

run_encode_prune(){
    local chunk_size=$1
    local encode_memory=$2
    local encode_window=$3
    local prune_temporal=$4
    local prune_spatial=$5
    local name="C${chunk_size}_encode_M${encode_memory}_W${encode_window}_PT${prune_temporal}_PS${prune_spatial}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py \
    --mode "encode" --chunk_size $chunk_size --encode_memory $encode_memory --encode_window $encode_window \
    --plot_file "../results/plots/${name}.png"   \
    --prune --prune_temporal $prune_temporal --prune_spatial $prune_spatial \
    > ../results/logs/${name}.log 2>&1
}

run_decode_prune(){
    local chunk_size=$1
    local decode_select=$2
    local encode_window=$3
    local prune_temporal=$4
    local prune_spatial=$5
    local name="C${chunk_size}_decode_S${decode_select}_W${encode_window}_PT${prune_temporal}_PS${prune_spatial}"
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py \
    --mode "decode" --chunk_size $chunk_size --decode_select $decode_select --encode_window $encode_window\
    --plot_file "../results/plots/${name}.png"   \
    > ../results/logs/${name}.log 2>&1
}

run_encode_encrypt 1 256 64
run_decode_decrypt 1 16 64
run_decode_decrypt 1 32 64
run_decode_decrypt 1 64 64
run_decode_decrypt 1 128 64
run_decode_decrypt 1 0 64

run_encode_encrypt 4 64 16
run_decode_decrypt 4 4 16
run_decode_decrypt 4 8 16
run_decode_decrypt 4 16 16
run_decode_decrypt 4 32 16
run_decode_decrypt 4 0 16


#run_encode_encrypt 16 16 4
#run_decode_decrypt 16 1 4
#run_decode_decrypt 16 2 4
#run_decode_decrypt 16 4 4
#run_decode_decrypt 16 8 4
#run_decode_decrypt 16 0 4