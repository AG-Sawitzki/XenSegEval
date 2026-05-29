# 10xSegEval
Does Seg and does Eval.
soon.

# Environments
Always install cuda and torch separately. Tensorflow propably too.

cpsam: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
dissect: 
    - pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    - python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
ucs: for me worked with env-file. but if not:
    - micromamba install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# proseg
start_proseg.sh has the flag --overwrite (L22 & L31) active. if you do not want old files to be overwritten then remove this line.