# Need to do custom install to prevent dependency errors
conda create -y --name sister python=3.8
source activate sister

conda install gdal -y

git clone https://github.com/EnSpec/hytools.git -b 1.5.0
cd hytools
pip install .

pip install ray==2.2.0
