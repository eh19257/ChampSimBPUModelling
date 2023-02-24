ENDIAN = "big"

file = open("./bp_models/2_bit", "rb")

count = 0
while True:
    ip  = int.from_bytes(file.read(8), ENDIAN)
    b_a = int.from_bytes(file.read(8), ENDIAN)
    b_t = int.from_bytes(file.read(1), ENDIAN)

    print("IP: 0x{0:16x}, branch_target: 0x{1:16x}, branch_type: 0x{2:d}".format(ip, b_a, b_t))
    count += 1
