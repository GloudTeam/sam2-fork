# Fine tuning

- VM :

  - GCP 에서 instance setup 필요
    - `NVIDIA A100 80GB` 사용했었음
    - Storage 는 그냥 넉넉하게 300~500GB
    - OS : `debian-12-bookworm-v20250910`
    - GPU driver 설치 :
      - https://cloud.google.com/compute/docs/gpus/install-drivers-gpu?hl=ko

- 환경 셑업

  - install conda
    - ```
      cd ~
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
      ./Miniconda3-latest-Linux-x86_64.sh
      ```
  - create conda env
    - `conda create -n sam python=3.11 `
  - install dependencies
    - ```
      conda activate sam
      pip install -e ".[dev]"
      ```

- dataset 준비하기

  - 참조 preprocessing 코드 :

    - https://github.com/GloudTeam/veneer-editor/blob/main/apps/python-server/src/veneer_contrib/jhoon/preprocessing/preprocess_tgn_for_sam.py

  - 구조 어떤식으로 넣었는지
    - 예시가 궁금하시다면 : `gs://beluai-experiments/veneer/sam_tgn/250924_tgn_segment_every_teeth_individual` 참조 

    - ```
      preprocessed_dir
      |-- gt_folder
      |   |-- {case_id}
      |   |   |-- 11(tooth_number)
      |   |   |   |- 00000.png
      |   |   |-- 12
      |   |   |   |- 00000.png
      |   |   |-- 13
      |   |   |   |- 00000.png
      |   |-- 01328DDN
      |-- img_folder
      |   |-- {case_id}
      |   |   |-- 00000.jpg  (input image)
      |   |-- 01328DDN
      |   |   |-- 00000.jpg
      ```

- `.yaml` 파일 작성하기
  - 학습에 사용되는 각종 parameter 들을 hydra 로 instantiate
  - 참조 : https://github.com/GloudTeam/veneer-editor/blob/main/apps/python-server/src/veneer/models/sam2/configs/train_segment_every_teeth.yaml

- train command
  - ```
    cd sam2-fork
    python training/train.py -c 'configs/train_segment_every_teeth.yaml' --use-cluster 0 --num-gpus 1
    ```

  - 결과물이 `sam2-fork/sam2_logs/{yaml file name.yaml}` 에 저장됨 
    - 따라서 새로운 모델 학습할때마다 `.yaml` 파일 이름 새로 작성해줄 필요 **있음**.

- 현재 .yaml 파일에서는 5epoch 마다 체크포인트 저장하도록 되어있음.
- 저장된 `.pt` 파일을 `sam2_infer` 에서 사용하면서 테스트 가능