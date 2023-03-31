sudo apt-get update
sudo apt-get upgrade
sudo apt-get install gcc g++ build-essential python3 python3-pip make

# pip3 install pandas numpy torch

echo "export PATH=/vagrant/tracer/pin/pin-3.22-98547-g7a303a835-gcc-linux:$PATH" >> ~/.bashrc
echo "export TRACER=/home/eh19257/Uni/FourthYear/ip/ChampSimBPUModelling/tracer/pin/obj-intel64/champsim_tracer.so" >> ~/.bashrc

source ~/.bashrc
