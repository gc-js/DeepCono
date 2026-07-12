# esm2_base_model
wget -O ./esm2_base_model/esm2_t6_8M_UR50D/pytorch_model.bin "https://zenodo.org/records/21318628/files/esm2_t6_8M_UR50D_pytorch_model.bin?download=1"
wget -O ./esm2_base_model/esm2_t12_35M_UR50D/pytorch_model.bin "https://zenodo.org/records/21318628/files/esm2_t12_35M_UR50D_pytorch_model.bin?download=1"
wget -O ./esm2_base_model/esm2_t30_150M_UR50D/pytorch_model.bin "https://zenodo.org/records/21318628/files/esm2_t30_150M_UR50D_pytorch_model.bin?download=1"
# 1_MLM_lora
wget -O ./MLM_lora/model/mlm_conoserver_lora_esm2_t30_150M_UR50D/adapter_model.safetensors "https://zenodo.org/records/21318628/files/adapter_model.safetensors?download=1"
# 2_conopep_ident
wget -O ./conopep_ident/model/conopep_ident_best_model.pth "https://zenodo.org/records/21318628/files/conopep_ident_best_model.pth?download=1"
# 2_ctx_ident
wget -O ./ctx_ident/model/ctx_ident_best_model.pth "https://zenodo.org/records/21318628/files/ctx_ident_best_model.pth?download=1"
# 2_ctx_target
wget -O ./ctx_target/model/ctx_target_best_model.pth "https://zenodo.org/records/21318628/files/ctx_target_best_model.pth?download=1"