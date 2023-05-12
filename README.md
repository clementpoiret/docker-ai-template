# Setup

Just copy the content of `sample_data` to the S3 bucket you want to mount to `/workspace/data`.

# Usage

```sh
ovhai job run \                                                                             base
--name docker-demo \
--flavor ai1-1-gpu \
--gpu 1 \
--volume input@GRA/:/workspace/data:RO \
--volume output_charly@GRA/:/workspace/models:RWD \
registry.test.com/docker-demo/main:latest \
-- \
python train.py \
--batch_size 4 \
--lr 5e-4 \
--epochs 64
```
