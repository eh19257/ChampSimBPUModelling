export PATH_TO_SPEC="/vagrant/benchmarks/spec-cpu-2017/build"
#export TRACES="/vagrant/traces"
#export CHAMPSIM="/vagrant" 
#export PIN_ROOT="/vagrant/tracer/pin/pin-3.26-98690-g1fc9d60e6-gcc-linux"


# EDIT the variable BP here - then run the program
export BP="2_bit"
export TRACES="/vagrant/traces"
export CHAMPSIM="/vagrant"
./config.sh config_${BP}.json
make
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/502.GCC.bin ${TRACES}/champsim-502.GCC.xz
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/525.X264.bin ${TRACES}/champsim-525.X264.xz
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/541.LEELA.bin ${TRACES}/champsim-541.LEELA.xz
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/523.XALANCBMK.bin ${TRACES}/champsim-523.XALANCBMK.xz
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/531.DEEPSJENG.bin ${TRACES}/champsim-531.DEEPSJENG.xz
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/557_XZ.bin ${TRACES}/champsim-557_XZ.xz
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/505.MCF.bin ${TRACES}/champsim-505.MCF.xz
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/pwd.bin ${TRACES}/champsim-pwd.xz

scp ${CHAMPSIM}/data/bp_models/${BP}/502.GCC.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/
scp ${CHAMPSIM}/data/bp_models/${BP}/525.X264.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/
scp ${CHAMPSIM}/data/bp_models/${BP}/541.LEELA.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/
scp ${CHAMPSIM}/data/bp_models/${BP}/523.XALANCBMK.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/
scp ${CHAMPSIM}/data/bp_models/${BP}/531.DEEPSJENG.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/
scp ${CHAMPSIM}/data/bp_models/${BP}/557_XZ.bi bp1-copy:/user/work/eh19257/data/bp_models/${BP}/
scp ${CHAMPSIM}/data/bp_models/${BP}/505.MCF.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/
scp ${CHAMPSIM}/data/bp_models/${BP}/pwd.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/