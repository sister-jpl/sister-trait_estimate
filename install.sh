# Need to do custom install to prevent dependency errors
conda create -y --name sister python=3.8
source activate sister

git clone https://github.com/EnSpec/hytools.git
cd hytools
pip install .
