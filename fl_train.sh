rds=(0 1 2 3 4 5 6 7 8 9 10)
clients=(0 1 2 3 4)
for rd in "${rds[@]}"
  do
    for client in "${clients[@]}"
      do
        python client_train.py --config_path ./config --config_files fl/kc_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama1b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd" --client "$client"
    done
    python server_train.py --config_path ./config --config_files fl/kc_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama1b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd"
done
for rd in "${rds[@]}"
  do
    for client in "${clients[@]}"
      do
        python client_train.py --config_path ./config --config_files fl/kmeans_kc_llama.yaml data/fl_kmeans_5_alpaca_gpt4.yaml model/llama1b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd" --client "$client"
    done
    python server_train.py --config_path ./config --config_files fl/kmeans_kc_llama.yaml data/fl_kmeans_5_alpaca_gpt4.yaml model/llama1b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd"
done

#for rd in "${rds[@]}"
#  do
#    for client in "${clients[@]}"
#      do
#        python client_train.py --config_path ./config --config_files fl/kc_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama8b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd" --client "$client"
#    done
#    python server_train.py --config_path ./config --config_files fl/kc_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama8b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd"
#done

rds=(0)
clients=(0 1 2 3 4)
for rd in "${rds[@]}"
  do
    for client in "${clients[@]}"
      do
        python client_train.py --config_path ./config --config_files fl/all_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama1b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd" --client "$client"
    done
    python server_train.py --config_path ./config --config_files fl/all_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama1b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd"
done
#for rd in "${rds[@]}"
#  do
#    for client in "${clients[@]}"
#      do
#        python client_train.py --config_path ./config --config_files fl/all_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama8b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd" --client "$client"
#    done
#    python server_train.py --config_path ./config --config_files fl/all_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama8b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd"
#done

base_model_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/model/meta-llama/Llama-3.2-1B"
eval_base_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/user/zyk/projects/FedDS/output/evol_res/meta-llama/Llama-3.2-1B/"
eval_opt_base_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/user/zyk/projects/FedDS/evol_answer/meta-llama/Llama-3.2-1B"
rds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
model_paths=("fl_random_kc_alpaca_lora" "fl_random_all_alpaca_lora" "fl_kmeans_kc_alpaca_lora")
tasks=("mmlu" "winogrande" "arc_challenge" "hellaswag" "truthfulqa_mc2" "ai2_arc")
for model_path in "${model_paths[@]}"
  do
    for rd in "${rds[@]}"
      do
        for task in "${tasks[@]}"
          do
            pth="${eval_base_path}${model_path}/output/server/model/rd_${rd}/"
            output_pth="${eval_opt_base_path}/${task}/${model_path}/rd_${rd}/"
            echo "testing $pth"
            echo "output to $output_pth"
            HF_ENDPOINT=https://hf-mirror.com /home/smilelabfd/.local/bin/lm-eval --model hf \
                    --model_args pretrained="$base_model_path",peft="$pth" \
                    --device cuda:0 \
                    --tasks "$task" \
                    --batch_size auto \
                    --output_path "$output_pth"
        done
    done
done

rds=(0 1 2 3 4 5 6 7 8 9 10)
clients=(0 1 2 3 4)
for rd in "${rds[@]}"
  do
    for client in "${clients[@]}"
      do
        python client_train.py --config_path ./config --config_files fl/kc_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama3b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd" --client "$client"
    done
    python server_train.py --config_path ./config --config_files fl/kc_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama3b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd"
done
for rd in "${rds[@]}"
  do
    for client in "${clients[@]}"
      do
        python client_train.py --config_path ./config --config_files fl/kmeans_kc_llama.yaml data/fl_kmeans_5_alpaca_gpt4.yaml model/llama3b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd" --client "$client"
    done
    python server_train.py --config_path ./config --config_files fl/kmeans_kc_llama.yaml data/fl_kmeans_5_alpaca_gpt4.yaml model/llama3b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd"
done
rds=(0)
clients=(0 1 2 3 4)
for rd in "${rds[@]}"
  do
    for client in "${clients[@]}"
      do
        python client_train.py --config_path ./config --config_files fl/all_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama3b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd" --client "$client"
    done
    python server_train.py --config_path ./config --config_files fl/all_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama3b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd"
done

base_model_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/model/meta-llama/Llama-3.2-3B"
eval_base_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/user/zyk/projects/FedDS/output/evol_res/meta-llama/Llama-3.2-3B/"
eval_opt_base_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/user/zyk/projects/FedDS/evol_answer/meta-llama/Llama-3.2-3B"
rds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
model_paths=("fl_random_kc_alpaca_lora" "fl_random_all_alpaca_lora" "fl_kmeans_kc_alpaca_lora")
tasks=("mmlu" "winogrande" "arc_challenge" "hellaswag" "truthfulqa_mc2" "ai2_arc")
for model_path in "${model_paths[@]}"
  do
    for rd in "${rds[@]}"
      do
        for task in "${tasks[@]}"
          do
            pth="${eval_base_path}${model_path}/output/server/model/rd_${rd}/"
            output_pth="${eval_opt_base_path}/${task}/${model_path}/rd_${rd}/"
            echo "testing $pth"
            echo "output to $output_pth"
            HF_ENDPOINT=https://hf-mirror.com /home/smilelabfd/.local/bin/lm-eval --model hf \
                    --model_args pretrained="$base_model_path",peft="$pth" \
                    --device cuda:0 \
                    --tasks "$task" \
                    --batch_size auto \
                    --output_path "$output_pth"
        done
    done
done


rds=(0 1 2 3 4 5 6 7 8 9 10)
clients=(0 1 2 3 4)
for rd in "${rds[@]}"
  do
    for client in "${clients[@]}"
      do
        python client_train.py --config_path ./config --config_files fl/kc_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama8b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd" --client "$client"
    done
    python server_train.py --config_path ./config --config_files fl/kc_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama8b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd"
done
for rd in "${rds[@]}"
  do
    for client in "${clients[@]}"
      do
        python client_train.py --config_path ./config --config_files fl/kmeans_kc_llama.yaml data/fl_kmeans_5_alpaca_gpt4.yaml model/llama8b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd" --client "$client"
    done
    python server_train.py --config_path ./config --config_files fl/kmeans_kc_llama.yaml data/fl_kmeans_5_alpaca_gpt4.yaml model/llama8b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd"
done

rds=(0)
clients=(0 1 2 3 4)
for rd in "${rds[@]}"
  do
    for client in "${clients[@]}"
      do
        python client_train.py --config_path ./config --config_files fl/all_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama8b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd" --client "$client"
    done
    python server_train.py --config_path ./config --config_files fl/all_llama.yaml data/fl_5_alpaca_gpt4.yaml model/llama8b.yaml train/fl_train_config.yaml peft/lora.yaml --rd "$rd"
done

base_model_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/model/meta-llama/Llama-3.1-8B"
eval_base_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/user/zyk/projects/FedDS/output/evol_res/meta-llama/Llama-3.1-8B/"
eval_opt_base_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/user/zyk/projects/FedDS/evol_answer/meta-llama/Llama-3.1-8B"
rds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
model_paths=("fl_random_kc_alpaca_lora" "fl_random_all_alpaca_lora" "fl_kmeans_kc_alpaca_lora")
tasks=("mmlu" "winogrande" "arc_challenge" "hellaswag" "truthfulqa_mc2" "ai2_arc")
for model_path in "${model_paths[@]}"
  do
    for rd in "${rds[@]}"
      do
        for task in "${tasks[@]}"
          do
            pth="${eval_base_path}${model_path}/output/server/model/rd_${rd}/"
            output_pth="${eval_opt_base_path}/${task}/${model_path}/rd_${rd}/"
            echo "testing $pth"
            echo "output to $output_pth"
            HF_ENDPOINT=https://hf-mirror.com /home/smilelabfd/.local/bin/lm-eval --model hf \
                    --model_args pretrained="$base_model_path",peft="$pth" \
                    --device cuda:0 \
                    --tasks "$task" \
                    --batch_size auto \
                    --output_path "$output_pth"
        done
    done
done