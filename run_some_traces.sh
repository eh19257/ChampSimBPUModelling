export PATH_TO_SPEC="/vagrant/benchmarks/spec-cpu-2017/build"
#export TRACES="/vagrant/traces"
#export CHAMPSIM="/vagrant" 
#export PIN_ROOT="/vagrant/tracer/pin/pin-3.26-98690-g1fc9d60e6-gcc-linux"


# EDIT the variable BP here - then run the program
export BP="NT"
export TRACES="/vagrant/traces"
export CHAMPSIM="/vagrant"
./config.sh config_${BP}.json
make
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/502.GCC.bin ${TRACES}/champsim-502.GCC.xz > ${CHAMPSIM}/data/metrics/${BP}/502.GCC; echo "Done 502.GCC"
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/525.X264.bin ${TRACES}/champsim-525.X264.xz > ${CHAMPSIM}/data/metrics/${BP}/525.X264; echo "Done 525.X264"
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/541.LEELA.bin ${TRACES}/champsim-541.LEELA.xz > ${CHAMPSIM}/data/metrics/${BP}/541.LEELA; echo "Done 541.LEELA"
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/523.XALANCBMK.bin ${TRACES}/champsim-523.XALANCBMK.xz > ${CHAMPSIM}/data/metrics/${BP}/523.XALANCBMK; echo "Done 523.XALANCBMK"
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/531.DEEPSJENG.bin ${TRACES}/champsim-531.DEEPSJENG.xz > ${CHAMPSIM}/data/metrics/${BP}/531.DEEPSJENG; echo "Done 531.DEEPSJENG"
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/557_XZ.bin ${TRACES}/champsim-557_XZ.xz > ${CHAMPSIM}/data/metrics/${BP}/557_XZ; echo "Done 557_XZ"
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/505.MCF.bin ${TRACES}/champsim-505.MCF.xz > ${CHAMPSIM}/data/metrics/${BP}/505.MCF; echo "Done 505.MCF"
./bin/champsim -f ${CHAMPSIM}/data/bp_models/${BP}/pwd.bin ${TRACES}/champsim-pwd.xz > ${CHAMPSIM}/data/metrics/${BP}/pwd; echo "Done pwd"

echo "DONE! Gathered metrics for: ${BP}"