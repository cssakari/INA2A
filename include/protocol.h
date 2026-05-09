#pragma once

#include <cstdint>

#define MAGIC 0x494E4332  // "INC2"
constexpr uint16_t PROTOCOL_VERSION = 1;

enum MsgType {
    MSG_HELLO = 1,
    MSG_DISPATCH_TOKEN = 2,
    MSG_EXPERT_INPUT = 3,
    MSG_EXPERT_RESULT = 4,
    MSG_COMBINE_PULL_REQ = 5,
    MSG_COMBINE_RESULT = 6,
    MSG_NOT_READY = 7,
    MSG_ACK = 8,
    MSG_FINISH = 9
};

struct MsgHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t type;

    uint32_t header_len;
    uint32_t payload_len;

    uint64_t run_id;
    uint64_t microbatch_id;
    uint64_t token_id;

    uint32_t src_rank;
    uint32_t dst_rank;
    uint32_t origin_rank;

    uint64_t expert_bitmap;
    uint64_t global_idx;

    uint64_t timestamp_ns;
};

struct TokenDesc {
    uint64_t run_id;
    uint64_t microbatch_id;
    uint64_t token_id;
    uint32_t src_rank;
    uint32_t dst_rank;
    uint32_t origin_rank;
    uint64_t expert_bitmap;
    uint64_t global_idx;
    uint32_t payload_len;
};
