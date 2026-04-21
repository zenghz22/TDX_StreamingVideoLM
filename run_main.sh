run_encode(){
    local encode_memory=$1
    local encode_window=$2
    local name="encode_M${encode_memory}_W${encode_window}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --mode "encode" --encode_memory $encode_memory --encode_window $encode_window --plot_file "../results/plots/${name}.png" > ../results/logs/${name}.log 2>&1
}

run_decode(){
    local decode_select=$1
    local encode_window=$2
    local name="decode_S${decode_select}_W${encode_window}"
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --mode "decode" --decode_select $decode_select --encode_window $encode_window --plot_file "../results/plots/${name}.png"   > ../results/logs/${name}.log 2>&1
}

run_encode_decode(){
    local encode_memory=$1
    local encode_window=$2
    local decode_select=$3
    local name="encode_decode_M${encode_memory}_W${encode_window}_S${decode_select}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --mode "encode_decode" --encode_memory $encode_memory --encode_window $encode_window --decode_select $decode_select --plot_file "../results/plots/${name}.png"   > ../results/logs/${name}.log 2>&1
}


run_encode_encrypt(){
    local encode_memory=$1
    local encode_window=$2
    local name="encode_encrypt_M${encode_memory}_W${encode_window}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py \
    --mode "encode" --encode_memory $encode_memory --encode_window $encode_window \
    --plot_file "../results/plots/${name}.png"   \
    --encrypt \
    > ../results/logs/${name}.log 2>&1
}

run_decode_decrypt(){
    local decode_select=$1
    local encode_window=$2
    local name="decode_decrypt_S${decode_select}_W${encode_window}"
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py \
    --mode "decode" --decode_select $decode_select --encode_window $encode_window \
    --plot_file "../results/plots/${name}.png"   \
    --encrypt \
    > ../results/logs/${name}.log 2>&1
}

run_encode_prune(){
    local encode_memory=$1
    local encode_window=$2
    local prune_temporal=$3
    local prune_spatial=$4
    local name="encode_M${encode_memory}_W${encode_window}_PT${prune_temporal}_PS${prune_spatial}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py \
    --mode "encode" --encode_memory $encode_memory --encode_window $encode_window \
    --plot_file "../results/plots/${name}.png"   \
    --prune --prune_temporal $prune_temporal --prune_spatial $prune_spatial \
    > ../results/logs/${name}.log 2>&1
}

run_decode_prune(){
    local decode_select=$1
    local encode_window=$2
    local prune_temporal=$3
    local prune_spatial=$4
    local name="decode_S${decode_select}_W${encode_window}_PT${prune_temporal}_PS${prune_spatial}"
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py \
    --mode "decode" --decode_select $decode_select --encode_window $encode_window\
    --plot_file "../results/plots/${name}.png"   \
    > ../results/logs/${name}.log 2>&1
}

run_encode_encrypt 256 32
run_decode_decrypt 32 32
run_decode_decrypt 64 32