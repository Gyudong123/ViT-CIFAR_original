#!/bin/bash

# 파라미터 조합
heads=(1 2 4 8 12)
num_layers=(2 4 8)
hiddens=(48 96 192 384)
mlp_hiddens=(48 96 192 384)
batch_sizes=(64 128 256)
learning_rates=(1e-3 5e-4)


# 동시에 실행할 최대 실험 수
MAX_JOBS=3

# 각 조합에 대해 실험 실행
for head in "${heads[@]}"; do
    for num_layer in "${num_layers[@]}"; do
        for hidden in "${hiddens[@]}"; do
            for mlp_hidden in "${mlp_hiddens[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    for lr in "${learning_rates[@]}"; do
                        # 백그라운드 작업 수 확인 및 대기
                        while [ $(jobs | wc -l) -ge $MAX_JOBS ]; do
                            sleep 1
                        done

                        # 실험 실행
                        echo "Running experiment with head=$head, num_layer=$num_layer, hidden=$hidden, mlp_hidden=$mlp_hidden, batch_size = $batch_size, lr = $lr"
                        python main.py --head $head --num-layers $num_layer --hidden $hidden --mlp-hidden $mlp_hidden --batch-size $batch_size --lr $lr  &
                    done
                done
            done
        done
    done
done

# 모든 백그라운드 작업이 완료될 때까지 대기
wait
echo "All experiments completed."
