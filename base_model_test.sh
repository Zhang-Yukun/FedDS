sleep 20h
base_model_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/model/meta-llama/Llama-3.2-1B"
eval_opt_base_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/user/zyk/projects/FedDS/evol_answer/meta-llama/Llama-3.2-1B/"
rds=("0")
model_paths=("Llama-3.2-1B")
#tasks=("mmlu" "winogrande" "arc_challenge" "hellaswag" "truthfulqa_mc2" "ai2_arc")
tasks=("ai2_arc")
for model_path in "${model_paths[@]}"
  do
    for rd in "${rds[@]}"
      do
        for task in "${tasks[@]}"
          do
            pth=$base_model_path
            output_pth="${eval_opt_base_path}/${task}/${model_path}/rd_${rd}/"
            echo "testing $pth for $task"
            echo "output to $output_pth"
            HF_ENDPOINT=https://hf-mirror.com /home/smilelabfd/.local/bin/lm-eval --model hf \
                    --model_args pretrained="$base_model_path" \
                    --device cuda:0 \
                    --tasks "$task" \
                    --batch_size auto \
                    --output_path "$output_pth"
        done
    done
done

base_model_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/model/meta-llama/Llama-3.2-3B"
eval_opt_base_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/user/zyk/projects/FedDS/evol_answer/meta-llama/Llama-3.2-3B/"
rds=("0")
model_paths=("Llama-3.2-3B")
#tasks=("mmlu" "winogrande" "arc_challenge" "hellaswag" "truthfulqa_mc2" "ai2_arc")
tasks=("ai2_arc")
for model_path in "${model_paths[@]}"
  do
    for rd in "${rds[@]}"
      do
        for task in "${tasks[@]}"
          do
            pth=$base_model_path
            output_pth="${eval_opt_base_path}/${task}/${model_path}/rd_${rd}/"
            echo "testing $pth for $task"
            echo "output to $output_pth"
            HF_ENDPOINT=https://hf-mirror.com /home/smilelabfd/.local/bin/lm-eval --model hf \
                    --model_args pretrained="$base_model_path" \
                    --device cuda:0 \
                    --tasks "$task" \
                    --batch_size auto \
                    --output_path "$output_pth"
        done
    done
done

base_model_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/model/meta-llama/Llama-3.2-8B"
eval_opt_base_path="/cpfs01/projects-HDD/cfff-dea1e1ccd7cb_HDD/public/user/zyk/projects/FedDS/evol_answer/meta-llama/Llama-3.2-8B/"
rds=("0")
model_paths=("Llama-3.2-8B")
#tasks=("mmlu" "winogrande" "arc_challenge" "hellaswag" "truthfulqa_mc2" "ai2_arc")
tasks=("ai2_arc")
for model_path in "${model_paths[@]}"
  do
    for rd in "${rds[@]}"
      do
        for task in "${tasks[@]}"
          do
            pth=$base_model_path
            output_pth="${eval_opt_base_path}/${task}/${model_path}/rd_${rd}/"
            echo "testing $pth for $task"
            echo "output to $output_pth"
            HF_ENDPOINT=https://hf-mirror.com /home/smilelabfd/.local/bin/lm-eval --model hf \
                    --model_args pretrained="$base_model_path" \
                    --device cuda:0 \
                    --tasks "$task" \
                    --batch_size auto \
                    --output_path "$output_pth"
        done
    done
done