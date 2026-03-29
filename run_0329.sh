run(){
    local max_in_memory=$1
    local window_size=$2
    rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --max_in_memory $max_in_memory --window_size $window_size --plot_file "../data/plots/M${max_in_memory}_W${window_size}.png"  > ../data/logs/M${max_in_memory}_W${window_size}.log 2>&1
}

run 16 0
run 16 16
run 16 8
run 16 4
run 16 2
run 16 1

run 8 0
run 8 8
run 8 4
run 8 2
run 8 1

run 4 0
run 4 4
run 4 2
run 4 1

run 2 0
run 2 2
run 2 1

run 1 0
run 1 1
