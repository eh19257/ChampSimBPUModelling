#include "ooo_cpu.h"

uint8_t bit;

void O3_CPU::initialize_branch_predictor(){
    std::cout << "CPU " << cpu << " Static taken branch predictor" << std::endl;

    bit = 1;
}

uint8_t O3_CPU::predict_branch(uint64_t ip, uint64_t predicted_target, uint8_t always_taken, uint8_t branch_type){
    return bit >= 2;
}

void O3_CPU::last_branch_result(uint64_t ip, uint64_t branch_target, uint8_t taken, uint8_t branch_type){
    if (taken)  bit = std::min(bit + 1, 3);
    else        bit = std::max(bit - 1, 0);
}