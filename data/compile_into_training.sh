# EDIT the variable BP here - then run the program
export BP="gshare"

cat ./bp_models/${BP}/pwd.bin > ./bp_models/${BP}/train.bin
cat ./bp_models/${BP}/557_XZ.bin >> ./bp_models/${BP}/train.bin
cat ./bp_models/${BP}/531.DEEPSJENG.bin >> ./bp_models/${BP}/train.bin
cat ./bp_models/${BP}/523.XALANCBMK.bin >> ./bp_models/${BP}/train.bin

echo "./bp_models/${BP}/train.bin has been generated"