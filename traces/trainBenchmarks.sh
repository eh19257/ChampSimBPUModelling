export PATH_TO_SPEC=/vagrant/benchmarks/spec-cpu-2017/build

# 525.X264:
${PATH_TO_SPEC}/benchspec/CPU/525.x264_r/run/run_base_test_Training-m64.0000/x264_r_base.Training-m64 -o out ${PATH_TO_SPEC}/benchspec/CPU/525.x264_r/data/test/input/BuckBunny.264 1280x720

# EXCHANGE:
cd ${PATH_TO_SPEC}/benchspec/CPU/548.exchange2_r/run/run_base_train_Training-m64.0000
./exchange2_r_base.Training-m64 0

# XZ: 
${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/run/run_base_test_Training-m64.0000/xz_r_base.Training-m64 ${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/data/all/input/cpu2006docs.tar.xz 4 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1548636 1555348 0
${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/run/run_base_test_Training-m64.0000/xz_r_base.Training-m64 ${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/data/all/input/cpu2006docs.tar.xz 4 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1462248 -1 1
${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/run/run_base_test_Training-m64.0000/xz_r_base.Training-m64 ${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/data/all/input/cpu2006docs.tar.xz 4 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1428548 -1 2
${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/run/run_base_test_Training-m64.0000/xz_r_base.Training-m64 ${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/data/all/input/cpu2006docs.tar.xz 4 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1034828 -1 3e
${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/run/run_base_test_Training-m64.0000/xz_r_base.Training-m64 ${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/data/all/input/cpu2006docs.tar.xz 4 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1061968 -1 4
${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/run/run_base_test_Training-m64.0000/xz_r_base.Training-m64 ${PATH_TO_SPEC}/benchspec/CPU/557.xz_r/data/all/input/cpu2006docs.tar.xz 4 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1034588 -1 4e


#502.GCC:
${PATH_TO_SPEC}/benchspec/CPU/502.gcc_r/run/run_base_test_Training-m64.0000/cpugcc_r_base.Training-m64 -fno-strict-aliasing -O3 -finline-limit=50000 ${PATH_TO_SPEC}/benchspec/CPU/502.gcc_r/data/test/input/t1.c