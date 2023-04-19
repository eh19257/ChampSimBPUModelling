# EDIT the variable BP here - then run the program
export BP="BT_FNT"

cat ./bp_models/${BP}/pwd.bin > train.bin
cat ./bp_models/${BP}/557_XZ.bin >> train.bin
cat ./bp_models/${BP}/531.DEEPSJENG.bin >> train.bin
cat ./bp_models/${BP}/523.XALANCBMK.bin >> train.bin

echo "./bp_models/${BP}/train.bin has been generated"