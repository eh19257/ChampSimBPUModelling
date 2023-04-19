# EDIT the variable BP here - then run the program
export BP="BT_FNT"

cat ./data/bp_models/${BP}/pwd.bin > train.bin
cat ./data/bp_models/${BP}/557_XZ.bin >> train.bin
cat ./data/bp_models/${BP}/531.DEEPSJENG.bin >> train.bin
cat ./data/bp_models/${BP}/523.XALANCBMK.bin >> train.bin

echo "./data/bp_models/${BP}/train.bin has been generated"