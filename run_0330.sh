run_decode_indiced(){
    local max_in_memory=$1
    local window_size=$2
    local decode_indices=$3
    local name="test_M${max_in_memory}_W${window_size}_D${decode_indices}"
    #rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --max_in_memory $max_in_memory --window_size $window_size --plot_file "../data/plots/${name}.png" --decode_indices $decode_indices  > ../data/logs/${name}.log 2>&1
}

run_decode_full(){
    local max_in_memory=$1
    local window_size=$2
    local decode_indices=$3
    local name="test_M${max_in_memory}_W${window_size}_Dfull"
    #rm -rf ../data/kv_cache_chunks/
    HF_ENDPOINT='https://hf-mirror.com' python main_td.py --max_in_memory $max_in_memory --window_size $window_size --plot_file "../data/plots/${name}.png"  > ../data/logs/${name}.log 2>&1
}

run_decode_full 16 0
run_decode_indiced 16 0 "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
run_decode_indiced 16 0 "0,1,2,3,4,5,6,7,8,9,10,11,12,13"
run_decode_indiced 16 0 "0,1,2,3,4,5,6,7,8,9,10,11,12"
run_decode_indiced 16 0 "0,1,2,3,4,5,6,7,8,9,10,11"
run_decode_indiced 16 0 "0,1,2,3,4,5,6,7,8,9,10"
run_decode_indiced 16 0 "0,1,2,3,4,5,6,7,8,9"
run_decode_indiced 16 0 "0,1,2,3,4,5,6,7,8"
run_decode_indiced 16 0 "0,1,2,3,4,5,6,7"
run_decode_indiced 16 0 "0,1,2,3,4,5,6"
run_decode_indiced 16 0 "0,1,2,3,4,5"
run_decode_indiced 16 0 "0,1,2,3,4"
run_decode_indiced 16 0 "0,1,2,3"
run_decode_indiced 16 0 "0,1,2"
run_decode_indiced 16 0 "0,1"
run_decode_indiced 16 0 "0"