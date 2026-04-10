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
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --mode "decode" --decode_select $decode_select --plot_file "../results/plots/${name}.png"   > ../results/logs/${name}.log 2>&1
}

run_encode_decode(){
    local encode_memory=$1
    local encode_window=$2
    local decode_select=$3
    local name="encode_decode_M${encode_memory}_W${encode_window}_S${decode_select}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --mode "encode_decode" --encode_memory $encode_memory --encode_window $encode_window --decode_select $decode_select --plot_file "../results/plots/${name}.png"   > ../results/logs/${name}.log 2>&1
}

run_encode 16 8
run_decode 1 0
run_decode 2 0
run_decode 4 0
run_decode 8 0
run_decode 12 0
run_decode 16 0
run_decode 0 0