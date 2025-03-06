runai submit cls \
    --image jonathancosme/base-notebook-root \
    --gpu 0.3 \
    --node-type A100 \
    --cpu 8 \
    --cpu-limit 8 \
    --memory-limit 25G \
    --project vocim-xiaoran \
    --large-shm \
    --preemptible \
    -- /bin/bash /mydata/vocim/xiaoran/scripts/bird_identity_classification/train.sh

runai submit eval \
    --image jonathancosme/base-notebook-root \
    --gpu 0.2 \
    --node-type A100 \
    --cpu 8 \
    --cpu-limit 8 \
    --memory-limit 25G \
    --project vocim-xiaoran \
    --large-shm \
    --preemptible \
    -- /bin/bash /mydata/vocim/xiaoran/scripts/bird_identity_classification/eval.sh

python train.py --train_json_data /mydata/vocim/zachary/data/cropped_merged_annotations.json \
                --eval_json_data /mydata/vocim/zachary/data/cropped_merged_annotations.json \
                --img_dir /mydata/vocim/zachary/data/cropped

python train.py --train_json_data /mydata/vocim/zachary/data/cropped_merged_annotations.json \
                --eval_json_data /mydata/vocim/zachary/data/cropped_merged_annotations.json \
                --img_dir /mydata/vocim/zachary/data/cropped 2>&1 | tee training.log


runai submit job1 -i zacharyydoll/gpu_image:latest \
                  --gpu 0.3 \
                  --node-type A100 \
                  --cpu 8 \
                  --memory-limit 25G \
                  --interactive \
                  --attach
