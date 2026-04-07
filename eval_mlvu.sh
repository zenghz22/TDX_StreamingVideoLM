run_encode(){
    local anon_index=$1
    local encode_memory=$2
    local encode_window=$3
    local name="mlvu_${anon_index}_encode_M${encode_memory}_W${encode_window}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python eval_mlvu.py --anon_index $anon_index --mode "encode" --encode_memory $encode_memory --encode_window $encode_window --plot_file "../data/plots/${name}.png" > ../data/logs/${name}.log 2>&1
}

run_decode(){
    local anon_index=$1
    local decode_select=$2
    local encode_window=$3
    local name="mlvu_${anon_index}_dncode_S${decode_select}_W${encode_window}"
    HF_ENDPOINT='https://hf-mirror.com' python eval_mlvu.py --anon_index $anon_index --mode "decode" --decode_select $decode_select --plot_file "../data/plots/${name}.png"   > ../data/logs/${name}.log 2>&1
}


for anon_index in {0..9}; do
    for encode_window in 4 8 16; do
        run_encode $anon_index 64 $encode_window
        for decode_select in 4 8 16 0; do
            run_decode $anon_index $decode_select $encode_window
        done
    done
done