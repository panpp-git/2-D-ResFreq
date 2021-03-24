# 2-D ResFreq
Codes for Deep Learning-based 2-D Frequency Estimation of Multiple Sinusoidals

This repo. includes:
--Training codes for 2D-ResFreq.(train_2D_new3.py)
--Experiments in the paper.
1. FNR_MOTE.py for FNR of multiple sinusoidals
2. Mainlobe.py for the comparison of mainlobe and sidelobe
3. ACCURACY_MOTE.py for the RMSE of frequency estimation for a single component
4. RESOLUTION_MOTE.py for the resolution performance
5. FNR_MOTE_AMP_MOTE.py for the MSE of amplitude estimation for multiple sinusoidals


--Requirements required for running the codes.
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: win-64
absl-py=0.11.0=py36haa95532_0
aiohttp=3.7.2=py36h2bbff1b_1
async-timeout=3.0.1=py36_0
attrs=20.2.0=py_0
blas=1.0=mkl
blinker=1.4=py36_0
brotlipy=0.7.0=py36he774522_1000
ca-certificates=2020.10.14=0
cachetools=4.1.1=py_0
certifi=2020.6.20=pyhd3eb1b0_3
cffi=1.14.3=py36h7a1dbc1_0
chardet=3.0.4=py36_1003
click=7.1.2=py_0
cryptography=3.1.1=py36h7a1dbc1_0
cudatoolkit=10.0.130=0
cycler=0.10.0=py36h009560c_0
decorator=4.4.1=pypi_0
flask=0.12.2=py36_0
freetype=2.10.4=hd328e21_0
future=0.16.0=py36_1
gevent=1.4.0=py36he774522_0
google-auth=1.23.0=pyhd3eb1b0_0
google-auth-oauthlib=0.4.2=pyhd3eb1b0_2
greenlet=0.4.17=py36he774522_0
grpcio=1.31.0=py36he7da953_0
h5py=2.10.0=py36h5e291fa_0
hdf5=1.10.4=h7ebc959_0
icc_rt=2019.0.0=h0cc432a_1
icu=58.2=ha925a31_3
idna=2.10=py_0
idna_ssl=1.1.0=py36_0
imageio=2.6.1=pypi_0
importlib-metadata=2.0.0=py_1
intel-openmp=2020.2=254
itsdangerous=1.1.0=py36_0
jinja2=2.11.2=py_0
jpeg=9b=hb83a4c4_2
kiwisolver=1.3.0=py36hd77b12b_0
libpng=1.6.37=h2a8f88b_0
libprotobuf=3.13.0.1=h200bbdf_0
libtiff=4.1.0=h56a325e_1
lz4-c=1.9.2=hf4a77e7_3
markdown=3.3.2=py36_0
markupsafe=1.1.1=py36he774522_0
matlabengineforpython=R2019b=pypi_0
matplotlib=3.3.2=0
matplotlib-base=3.3.2=py36hba9282a_0
mkl=2020.2=256
mkl-service=2.3.0=py36hb782905_0
mkl_fft=1.2.0=py36h45dec08_0
mkl_random=1.1.1=py36h47e9c7a_0
mpmath=1.1.0=pypi_0
multidict=4.7.6=py36he774522_1
networkx=2.4=pypi_0
ninja=1.10.1=py36h7ef1ec2_0
numpy=1.19.2=py36hadc3359_0
numpy-base=1.19.2=py36ha3acd2a_0
oauthlib=3.1.0=py_0
olefile=0.46=py36_0
openssl=1.1.1h=he774522_0
pillow=8.0.1=py36h4fa10fc_0
pip=20.2.4=py36_0
protobuf=3.13.0.1=py36ha925a31_1
pyasn1=0.4.8=py_0
pyasn1-modules=0.2.8=py_0
pycparser=2.20=py_2
pyjwt=1.7.1=py36_0
pyopenssl=19.1.0=py_1
pyparsing=2.4.7=py_0
pyqt=5.9.2=py36h6538335_2
pyreadline=2.1=py36_1
pysocks=1.7.1=py36_0
python=3.6.9=h5500b2f_0
python-dateutil=2.8.1=py_0
pytorch=1.2.0=py3.6_cuda100_cudnn7_1
pywavelets=1.1.1=pypi_0
qt=5.9.7=vc14h73c81de_0
requests=2.24.0=py_0
requests-oauthlib=1.3.0=py_0
rsa=4.6=py_0
scikit-image=0.16.2=pypi_0
scipy=1.5.2=py36h9439919_0
setuptools=50.3.0=py36h9490d1a_1
sip=4.19.8=py36h6538335_0
six=1.15.0=py_0
sqlite=3.33.0=h2a8f88b_0
tensorboard=2.2.1=pyh532a8cf_0
tensorboard-plugin-wit=1.6.0=py_0
tk=8.6.10=he774522_0
torch-dct=0.1.5=pypi_0
torchvision=0.4.0=py36_cu100
tornado=6.0.4=py36he774522_1
tqdm=4.50.2=pyh9f0ad1d_0
typing-extensions=3.7.4.3=0
typing_extensions=3.7.4.3=py_0
urllib3=1.25.11=py_0
vc=14.1=h0510ff6_4
vs2015_runtime=14.16.27012=hf0eaf9b_3
werkzeug=1.0.1=py_0
wheel=0.35.1=py_0
win_inet_pton=1.1.0=py36_0
wincertstore=0.2=py36h7fe50ca_0
xz=5.2.5=h62dcd97_0
yarl=1.5.1=py36he774522_0
zipp=3.4.0=pyhd3eb1b0_0
zlib=1.2.11=h62dcd97_4
zstd=1.4.5=h04227a9_0

