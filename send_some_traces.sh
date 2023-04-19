export PATH_TO_SPEC="/vagrant/benchmarks/spec-cpu-2017/build"
#export TRACES="/vagrant/traces"
#export CHAMPSIM="/vagrant" 
#export PIN_ROOT="/vagrant/tracer/pin/pin-3.26-98690-g1fc9d60e6-gcc-linux"


# EDIT the variable BP here - then run the program
export BP="perceptron"
export CHAMPSIM="/home/eh19257/Uni/FourthYear/ip/ChampSimBPUModelling"
export TRACES=${CHAMPSIM}/traces

scp ${CHAMPSIM}/data/bp_models/${BP}/502.GCC.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/ && rm ${CHAMPSIM}/data/bp_models/${BP}/502.GCC.bin
scp ${CHAMPSIM}/data/bp_models/${BP}/525.X264.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/ && rm ${CHAMPSIM}/data/bp_models/${BP}/525.X264.bin
scp ${CHAMPSIM}/data/bp_models/${BP}/541.LEELA.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/ && rm ${CHAMPSIM}/data/bp_models/${BP}/541.LEELA.bin
scp ${CHAMPSIM}/data/bp_models/${BP}/523.XALANCBMK.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/ && rm ${CHAMPSIM}/data/bp_models/${BP}/523.XALANCBMK.bin
scp ${CHAMPSIM}/data/bp_models/${BP}/531.DEEPSJENG.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/ && rm ${CHAMPSIM}/data/bp_models/${BP}/531.DEEPSJENG.bin
scp ${CHAMPSIM}/data/bp_models/${BP}/557_XZ.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/ && rm ${CHAMPSIM}/data/bp_models/${BP}/557_XZ.bin
scp ${CHAMPSIM}/data/bp_models/${BP}/505.MCF.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/ && rm ${CHAMPSIM}/data/bp_models/${BP}/505.MCF.bin
scp ${CHAMPSIM}/data/bp_models/${BP}/pwd.bin bp1-copy:/user/work/eh19257/data/bp_models/${BP}/ && rm ${CHAMPSIM}/data/bp_models/${BP}/pwd.bin

echo "Sent and deleted with ${BP}"
#rm ${CHAMPSIM}/data/bp_models/${BP}/502.GCC.bin
#rm ${CHAMPSIM}/data/bp_models/${BP}/502.GCC.bin
