export PATH_TO_SPEC="/vagrant/benchmarks/spec-cpu-2017/build"
export TRACES="/vagrant/traces"
export CHAMPSIM="/vagrant" 
export PIN_ROOT="/vagrant/tracer/pin/pin-3.26-98690-g1fc9d60e6-gcc-linux"


# 525.X264:
#cd ${TRACES}
#$PIN_ROOT/pin -t ../tracer/pin/obj-intel64/champsim_tracer.so -o champsim-525.X264 -- ${PATH_TO_SPEC}/benchspec/CPU/525.x264_r/run/run_base_test_Training-m64.0000/x264_r_base.Training-m64 -o out ${PATH_TO_SPEC}/benchspec/CPU/525.x264_r/data/test/input/BuckBunny.264 1280x720
#xz champsim-525.X264

#export BIN_FILE=525.X264.bin
#export XZ_FILE=champsim-525.X264.xz

# 502.GCC:
cd ${TRACES}
$PIN_ROOT/pin -t ../tracer/pin/obj-intel64/champsim_tracer.so -o champsim-502.GCC -- ${PATH_TO_SPEC}/benchspec/CPU/502.gcc_r/run/run_base_test_Training-m64.0000/cpugcc_r_base.Training-m64 -fno-strict-aliasing -O3 -finline-limit=50000 ${PATH_TO_SPEC}/benchspec/CPU/502.gcc_r/data/test/input/t1.c
xz champsim-502.GCC

export BIN_FILE=502.GCC.bin
export XZ_FILE=champsim-502.GCC.xz

chmod 777 ${XZ_FILE}

echo ${BIN_FILE}
echo "2_bit"
cd ${CHAMPSIM}
./config.sh config_2_bit.json
make
./bin/champsim -f ${CHAMPSIM}/data/bp_models/2_bit/${BIN_FILE} ${TRACES}/${XZ_FILE}
scp ${BIN_PATH} bp1-copy:/user/work/eh19257/data/bp_models/2_bit/
rm ${CHAMPSIM}/data/bp_models/2_bit/${BIN_FILE}

echo "AT"
./config.sh config_AT.json
make
./bin/champsim -f ${CHAMPSIM}/data/bp_models/AT/${BIN_FILE} ${TRACES}/${XZ_FILE}
scp ${BIN_PATH} bp1-copy:/user/work/eh19257/data/bp_models/AT/
rm ${CHAMPSIM}/data/bp_models/AT/${BIN_FILE}

echo "BT_FNT"
./config.sh config_BT_FNT.json
make
./bin/champsim -f ${CHAMPSIM}/data/bp_models/BT_FNT/${BIN_FILE} ${TRACES}/${XZ_FILE}
scp ${BIN_PATH} bp1-copy:/user/work/eh19257/data/bp_models/BT_FNT/
rm ${CHAMPSIM}/data/bp_models/BT_FNT/${BIN_FILE}

echo "GSHARE"
./config.sh config_gshare.json
make
./bin/champsim -f ${CHAMPSIM}/data/bp_models/gshare/${BIN_FILE} ${TRACES}/${XZ_FILE}
scp ${BIN_PATH} bp1-copy:/user/work/eh19257/data/bp_models/gshare/
rm ${CHAMPSIM}/data/bp_models/gshare/${BIN_FILE}

echo "NT"
./config.sh config_NT.json
make
./bin/champsim -f ${CHAMPSIM}/data/bp_models/NT/${BIN_FILE} ${TRACES}/${XZ_FILE}
scp ${BIN_PATH} bp1-copy:/user/work/eh19257/data/bp_models/NT/
rm ${CHAMPSIM}/data/bp_models/NT/${BIN_FILE}

echo "PERCEPTRON"
./config.sh config_perceptron.json
make
./bin/champsim -f ${CHAMPSIM}/data/bp_models/perceptron/${BIN_FILE} ${TRACES}/${XZ_FILE}
scp ${BIN_PATH} bp1-copy:/user/work/eh19257/data/bp_models/perceptron/
rm ${CHAMPSIM}/data/bp_models/perceptron/${BIN_FILE}

rm ${TRACES}/${XZ_FILE}


# 557.XZ: 
#$PIN_ROOT/pin -t ../tracer/pin/obj-intel64/champsim_tracer.so -o champsim-557_XZ -- ${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/run/run_base_test_Training-m64.0000/xz_r_base.Training-m64 ${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/data/all/input/cpu2006docs.tar.xz 4 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1548636 1555348 0
#xz champsim-557_XZ
#mv champsim-557_XZ.xz champsim-557.XZ.xz

# 502.GCC:
#$PIN_ROOT/pin -t ../tracer/pin/obj-intel64/champsim_tracer.so -o champsim-502.GCC -- ${PATH_TO_SPEC}/benchspec/CPU/502.gcc_r/run/run_base_test_Training-m64.0000/cpugcc_r_base.Training-m64 -fno-strict-aliasing -O3 -finline-limit=50000 ${PATH_TO_SPEC}/benchspec/CPU/502.gcc_r/data/test/input/t1.c
#xz champsim-502.GCC

# 505.MCF:
#$PIN_ROOT/pin -t ../tracer/pin/obj-intel64/champsim_tracer.so -o champsim-505.MCF -- ${PATH_TO_SPEC}/benchspec/CPU/505.mcf_r/run/run_base_test_Training-m64.0000/mcf_r_base.Training-m64 ${PATH_TO_SPEC}/benchspec/CPU/505.mcf_r/data/test/input/inp.in
#xz champsim-505.MCF

# 523.XALANCBMK:
#$PIN_ROOT/pin -t ../tracer/pin/obj-intel64/champsim_tracer.so -o champsim-523.XALANCBMK -- ${PATH_TO_SPEC}/benchspec/CPU/523.xalancbmk_r/run/run_base_test_Training-m64.0000/cpuxalan_r_base.Training-m64 ${PATH_TO_SPEC}/benchspec/CPU/523.xalancbmk_r/data/test/input/test.xml ${PATH_TO_SPEC}/benchspec/CPU/523.xalancbmk_r/data/test/input/xalanc.xsl
#xz champsim-523.XALANCBMK

# 531.DEEPSJENG:
#$PIN_ROOT/pin -t ../tracer/pin/obj-intel64/champsim_tracer.so -o champsim-531.DEEPSJENG -- ${PATH_TO_SPEC}/benchspec/CPU/531.deepsjeng_r/run/run_base_test_Training-m64.0000/deepsjeng_r_base.Training-m64 ${PATH_TO_SPEC}/benchspec/CPU/531.deepsjeng_r/data/test/input/test.txt
#xz champsim-531.DEEPSJENG

# 541.LEELA:
#$PIN_ROOT/pin -t ../tracer/pin/obj-intel64/champsim_tracer.so -o champsim-541.LEELA -- ${PATH_TO_SPEC}/benchspec/CPU/541.leela_r/run/run_base_test_Training-m64.0000/leela_r_base.Training-m64 ${PATH_TO_SPEC}/benchspec/CPU/541.leela_r/data/test/input/test.sgf
#xz champsim-541.LEELA

# EXCHANGE:
#cd ${PATH_TO_SPEC}/benchspec/CPU/548.exchange2_r/run/run_base_train_Training-m64.0000
#$PIN_ROOT/pin -t ../tracer/pin/obj-intel64/champsim_tracer.so -o champsim-548.EXCHANGE2 -- ./exchange2_r_base.Training-m64 0
#xz champsim-548.EXCHANGE2
#mv champsim-548.EXCHANGE2.xz /vagrant/traces/