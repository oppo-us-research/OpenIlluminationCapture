mkdir -p third_party/BackgroundMattingV2/bgmattingv2_models/
gdown 1zysR-jW6jydA2zkWfevxD1JpQHglKG1_ -O third_party/BackgroundMattingV2/bgmattingv2_models/
gdown 1b2FQH0yULaiBwe4ORUvSxXpdWLipjLsI -O third_party/BackgroundMattingV2/bgmattingv2_models/
gdown 1ErIAsB_miVhYL9GDlYUmfbqlV293mSYf -O third_party/BackgroundMattingV2/bgmattingv2_models/

pip install git+https://github.com/facebookresearch/segment-anything.git


mkdir -p third_party/segment_anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O third_party/segment_anything/sam_vit_h_4b8939.pth