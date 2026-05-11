#include "workload.h"

#include <algorithm>
#include <chrono>
#include <random>

uint64_t choose_topk_bitmap(int num_hosts, int topk) {
    if (num_hosts <= 0) {
        return 0;
    }
    if (topk <= 0) {
        topk = 1;
    }
    if (topk > num_hosts) {
        topk = num_hosts;
    }

    std::vector<int> indices(num_hosts);
    for (int i = 0; i < num_hosts; ++i) {
        indices[i] = i;
    }
    static std::mt19937_64 rng(static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::shuffle(indices.begin(), indices.end(), rng);

    uint64_t bitmap = 0;
    for (int i = 0; i < topk; ++i) {
        bitmap |= (1ull << indices[i]);
    }
    return bitmap;
}

TokenDesc make_token(uint64_t run_id, uint64_t mb, uint64_t token_id, int src_rank, int num_hosts, int topk) {
    TokenDesc t;
    t.run_id = run_id;
    t.microbatch_id = mb;
    t.token_id = token_id;
    t.src_rank = src_rank;
    t.dst_rank = 0;
    t.origin_rank = src_rank;
    t.global_idx = token_id;
    t.payload_len = 128 * 1024;
    t.expert_bitmap = choose_topk_bitmap(num_hosts, topk);
    return t;
}

std::vector<TokenDesc> make_workload(uint64_t run_id, int num_tokens, int src_rank, int num_hosts, int topk) {
    std::vector<TokenDesc> tokens;
    tokens.reserve(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        tokens.push_back(make_token(run_id, static_cast<uint64_t>(i), static_cast<uint64_t>(i), src_rank, num_hosts, topk));
    }
    return tokens;
}
