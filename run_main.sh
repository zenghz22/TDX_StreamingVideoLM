run_encode_decode(){
    local chunk_size=$1
    local encode_memory=$2
    local encode_window=$3
    local decode_select=$4
    local name="encode_decode_C${chunk_size}_M${encode_memory}_W${encode_window}_S${decode_select}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --mode "encode_decode" --chunk_size $chunk_size --encode_memory $encode_memory --encode_window $encode_window --decode_select $decode_select --plot_file "../results/plots/${name}.png"   > ../results/logs/${name}.log 2>&1
}

run_encode_encrypt(){
    local chunk_size=$1
    local encode_memory=$2
    local encode_window=$3
    local name="encode_encrypt_C${chunk_size}_M${encode_memory}_W${encode_window}"
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
    local name="decode_decrypt_C${chunk_size}_S${decode_select}_W${encode_window}"
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py \
    --mode "decode" --chunk_size $chunk_size --decode_select $decode_select --encode_window $encode_window \
    --plot_file "../results/plots/${name}.png"   \
    --encrypt \
    > ../results/logs/${name}.log 2>&1
}

memory_usage(){
    # 测量kvcache在磁盘中的占用
    # /home/zenghanzhang/tdx-streamvideo/data/kv_cache_chunks/kvpack.bin
    du /home/zenghanzhang/tdx-streamvideo/data/kv_cache_chunks/kvpack.bin
}

run_encode(){
    local chunk_size=$1
    local encode_memory=$2
    local encode_window=$3
    local name="encode_C${chunk_size}_M${encode_memory}_W${encode_window}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py \
    --mode "encode" \
    --chunk_size $chunk_size --encode_memory $encode_memory --encode_window $encode_window \
    --plot_file "../results/plots/${name}.png" > ../results/logs/${name}.log 2>&1
}

run_decode(){
    local chunk_size=$1
    local decode_select=$2
    local encode_window=$3
    local name="decode_C${chunk_size}_S${decode_select}_W${encode_window}"
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py \
    --mode "decode" \
    --chunk_size $chunk_size --decode_select $decode_select --encode_window $encode_window \
    --plot_file "../results/plots/${name}.png"   > ../results/logs/${name}.log 2>&1
}
# 基本功能验证
#run_encode 1 512 64
#run_decode 1 32 64
# 加密解密流程验证
#run_encode_encrypt 1 512 64
#run_decode 1 32 64
run_decode_decrypt 1 32 64
