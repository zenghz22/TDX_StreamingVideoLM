run_encode(){
    local anon_index=$1
    local encode_memory=$2
    local encode_window=$3
    local name="mlvu_${anon_index}_encode_M${encode_memory}_W${encode_window}"
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python eval_mlvu.py \
        --anon_index $anon_index \
        --mode "encode" \
        --encode_memory $encode_memory \
        --encode_window $encode_window \
        --plot_file "../results/plots/${name}.png" \
        > ../results/logs/${name}.log 2>&1
}

run_decode_all(){
    # 一次 Python 调用，模型只加载一次，跑所有 decode_select 设置
    local anon_index=$1
    local decode_select_list=$2   # 逗号分隔，如 "0,2,4,8,12"
    local encode_window=$3
    local name="mlvu_${anon_index}_decode_all_W${encode_window}"
    HF_ENDPOINT='https://hf-mirror.com' python eval_mlvu.py \
        --anon_index $anon_index \
        --mode "decode" \
        --decode_select "$decode_select_list" \
        --encode_window $encode_window \
        --plot_file "../results/plots/${name}.png" \
        > ../results/logs/${name}.log 2>&1
}

for encode_window in 128; do
    for anon_index in 1202 242 713 29 557; do
#    for anon_index in 1202 242 713 29 557 692 1025 133 686 1231; do
#    for anon_index in 130 359 409 474 507 516 687 1153 0 8; do
        #echo "Running encode mlvu_index=$anon_index, window=$encode_window"
        #run_encode $anon_index 256 $encode_window
        echo "Running all decode settings mlvu_index=$anon_index, window=$encode_window"
        run_decode_all $anon_index "16,32,64,128,256,384,0" $encode_window
    done
done