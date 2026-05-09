#pragma once

#include "protocol.h"

#include <cstdint>
#include <vector>

TokenDesc make_token(uint64_t run_id, uint64_t mb, uint64_t token_id, int src_rank, int num_hosts, int topk);
std::vector<TokenDesc> make_workload(uint64_t run_id, int num_tokens, int src_rank, int num_hosts, int topk);
uint64_t choose_topk_bitmap(int num_hosts, int topk);
