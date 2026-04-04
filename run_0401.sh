run_encode(){
    local encode_memory=$1
    local encode_window=$2
    local name="encode_M${encode_memory}_W${encode_window}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --mode "encode" --encode_memory $encode_memory --encode_window $encode_window --plot_file "../data/plots/${name}.png" > ../data/logs/${name}.log 2>&1
}

run_decode(){
    local decode_indices=$1
    local decode_select=$2
    local name="decode_I${decode_indices}_S${decode_select}"
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --mode "decode" --decode_indices $decode_indices --decode_select $decode_select --plot_file "../data/plots/${name}.png"   > ../data/logs/${name}.log 2>&1
}

run_encode_decode(){
    local encode_memory=$1
    local encode_window=$2
    local decode_indices=$3
    local decode_select=$4
    local name="encode_decode_M${encode_memory}_W${encode_window}_I${decode_indices}_S${decode_select}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --mode "encode_decode" --encode_memory $encode_memory --encode_window $encode_window --decode_indices $decode_indices --decode_select $decode_select --plot_file "../data/plots/${name}.png"   > ../data/logs/${name}.log 2>&1
}

#run_encode 16 4
run_decode "full" 0
run_decode "full" 2
run_decode "full" 4
run_decode "full" 8
run_decode "full" 12