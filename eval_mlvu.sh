run_encode_decode(){
    local encode_memory=$1
    local encode_window=$2
    local decode_select=$3
    local anon_index=$4
    local name="mlvu_zhz_${anon_index}_M${encode_memory}_W${encode_window}_S${decode_select}"
    #rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python eval_mlvu.py --mode "encode_decode" --encode_memory $encode_memory --encode_window $encode_window --decode_select $decode_select --plot_file "../data/plots/${name}.png"   >> ../data/logs/${name}.log 2>&1
}

run_encode_decode 256 8 "select8" 0