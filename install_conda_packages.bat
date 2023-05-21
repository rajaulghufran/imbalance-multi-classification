CALL %USERPROFILE%/anaconda3/Scripts/activate activate imbalance-multi-classification
CALL conda update -n base -c defaults conda -y
CALL conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
CALL conda install -c stanfordnlp -c conda-forge stanza -y
CALL conda install -c conda-forge scikit-learn -y
CALL conda install -c conda-forge scikit-learn-intelex -y
CALL conda install dpctl -c intel -y
CALL conda install dpcpp_cpp_rt -c intel -y
CALL conda install -c conda-forge matplotlib -y
CALL conda install -c conda-forge ipywidgets -y
CALL conda install -c conda-forge streamlit -y
CALL conda install -c plotly plotly -y