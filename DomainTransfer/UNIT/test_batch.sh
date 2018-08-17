python test_batch.py \
    --trainer UNIT \
    --config configs/unit_c2c6.yaml \
    --input_folder ../../Data/CarlaGen/C20_S1/RGB-1024-0.1/test \
    --output_folder outputs/unit_c2c6/predict_s50000/test \
    --checkpoint outputs/unit_c2c6/checkpoints/gen_00050000.pt \
    --a2b 1