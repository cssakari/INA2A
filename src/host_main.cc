#include "transport_softroce.h"
#include "workload.h"
#include "protocol.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <deque>

struct CombineKey {
    uint64_t run_id;
    uint64_t microbatch_id;
    uint64_t token_id;
    uint32_t origin_rank;
    uint64_t global_idx;

    bool operator<(const CombineKey& other) const {
        if (run_id != other.run_id) {
            return run_id < other.run_id;
        }
        if (microbatch_id != other.microbatch_id) {
            return microbatch_id < other.microbatch_id;
        }
        if (token_id != other.token_id) {
            return token_id < other.token_id;
        }
        if (origin_rank != other.origin_rank) {
            return origin_rank < other.origin_rank;
        }
        return global_idx < other.global_idx;
    }
};

struct CombineWaitState {
    bool ready = false;
    bool not_ready = false;
    MsgHeader header{};
    std::vector<char> payload;
};

struct ExpertTask {
    MsgHeader header{};
    std::vector<char> payload;
};

struct HostState {
    int rank = -1;
    SrConn* conn = nullptr;

    std::atomic<bool> running{true};

    std::mutex write_mutex;

    std::mutex state_mutex;
    std::condition_variable cv;

    std::map<CombineKey, bool> dispatch_acks;
    std::map<CombineKey, CombineWaitState> combine_states;

    std::mutex expert_mutex;
    std::condition_variable expert_cv;
    std::deque<ExpertTask> expert_queue;
};

static CombineKey make_key(const MsgHeader& header) {
    CombineKey key{};
    key.run_id = header.run_id;
    key.microbatch_id = header.microbatch_id;
    key.token_id = header.token_id;
    key.origin_rank = header.origin_rank;
    key.global_idx = header.global_idx;
    return key;
}

static bool send_to_switch(
    HostState& state,
    const MsgHeader& header,
    const void* payload
) {
    std::lock_guard<std::mutex> lock(state.write_mutex);
    return sr_write(state.conn, header, payload);
}

static void enqueue_expert_input(
    HostState& state,
    const MsgHeader& header,
    const std::vector<char>& payload
) {
    {
        std::lock_guard<std::mutex> lock(state.expert_mutex);

        ExpertTask task;
        task.header = header;
        task.payload = payload;

        state.expert_queue.push_back(std::move(task));
    }

    state.expert_cv.notify_one();
}

static void handle_expert_input(
    HostState& state,
    const MsgHeader& input_header,
    const std::vector<char>& payload
) {
    std::cout << "Host rank=" << state.rank
              << " processing EXPERT_INPUT token="
              << input_header.token_id
              << " origin_rank="
              << input_header.origin_rank
              << std::endl;

    std::vector<char> result(payload.size());

    /*
     * 这里模拟本地 expert MLP。
     * 你后面可以把这段替换成真正的 MLP forward。
     */
    for (size_t i = 0; i < payload.size(); ++i) {
        result[i] = static_cast<char>(
            (static_cast<uint8_t>(payload[i]) * 2) & 0xFF
        );
    }

    MsgHeader response{};
    response.magic = MAGIC;
    response.version = PROTOCOL_VERSION;
    response.type = MSG_EXPERT_RESULT;
    response.header_len = sizeof(MsgHeader);
    response.payload_len = static_cast<uint32_t>(result.size());

    response.run_id = input_header.run_id;
    response.microbatch_id = input_header.microbatch_id;
    response.token_id = input_header.token_id;

    response.src_rank = static_cast<uint32_t>(state.rank);
    response.dst_rank = 0;
    response.origin_rank = input_header.origin_rank;
    response.expert_bitmap = input_header.expert_bitmap;
    response.global_idx = input_header.global_idx;
    response.timestamp_ns = ns_timestamp();

    if (!send_to_switch(state, response, result.data())) {
        std::cerr << "Host rank=" << state.rank
                  << " failed to send EXPERT_RESULT token="
                  << input_header.token_id
                  << " from "
                  << input_header.origin_rank
                  << std::endl;

        state.running.store(false);
        state.cv.notify_all();
        state.expert_cv.notify_all();
        return;
    }

    std::cout << "Host rank=" << state.rank
              << " sent EXPERT_RESULT token="
              << input_header.token_id
              << " from "
              << input_header.origin_rank
              << std::endl;
}

static void expert_worker_loop(HostState* state) {
    while (true) {
        ExpertTask task;

        {
            std::unique_lock<std::mutex> lock(state->expert_mutex);

            state->expert_cv.wait(lock, [&]() {
                return !state->running.load() ||
                       !state->expert_queue.empty();
            });

            if (!state->running.load() && state->expert_queue.empty()) {
                break;
            }

            task = std::move(state->expert_queue.front());
            state->expert_queue.pop_front();
        }

        handle_expert_input(*state, task.header, task.payload);
    }

    std::cout << "Host rank=" << state->rank
              << " expert worker exited"
              << std::endl;
}

static void reader_loop(HostState* state) {
    while (state->running.load()) {
        MsgHeader header{};
        std::vector<char> payload;

        if (!sr_read(state->conn, header, payload)) {
            if (state->running.load()) {
                std::cerr << "Host rank=" << state->rank
                          << " sr_read failed"
                          << std::endl;
            }

            state->running.store(false);
            state->cv.notify_all();
            state->expert_cv.notify_all();
            break;
        }

        if (header.type == MSG_ACK) {
            CombineKey key = make_key(header);

            {
                std::lock_guard<std::mutex> lock(state->state_mutex);
                state->dispatch_acks[key] = true;
            }

            state->cv.notify_all();

            std::cout << "Host rank=" << state->rank
                      << " received ACK token="
                      << header.token_id
                      << std::endl;

            continue;
        }

        if (header.type == MSG_EXPERT_INPUT) {
            if (header.type == MSG_EXPERT_INPUT) {
                std::cout << "Host rank=" << state->rank
                    << " received EXPERT_INPUT token="
                    << header.token_id
                    << " origin_rank="
                    << header.origin_rank
                    << std::endl;

                enqueue_expert_input(*state, header, payload);
                continue;
            }
        }

        if (header.type == MSG_COMBINE_RESULT) {
            CombineKey key = make_key(header);

            {
                std::lock_guard<std::mutex> lock(state->state_mutex);

                auto& combine_state = state->combine_states[key];
                combine_state.ready = true;
                combine_state.not_ready = false;
                combine_state.header = header;
                combine_state.payload = payload;
            }

            state->cv.notify_all();

            std::cout << "Host rank=" << state->rank
                      << " received COMBINE_RESULT token="
                      << header.token_id
                      << " payload_len="
                      << header.payload_len
                      << std::endl;

            continue;
        }

        if (header.type == MSG_NOT_READY) {
            CombineKey key = make_key(header);

            {
                std::lock_guard<std::mutex> lock(state->state_mutex);

                auto& combine_state = state->combine_states[key];
                combine_state.not_ready = true;
                combine_state.ready = false;
                combine_state.header = header;
            }

            state->cv.notify_all();

            continue;
        }

        if (header.type == MSG_FINISH) {
            std::cout << "Host rank=" << state->rank
                      << " received FINISH from switch"
                      << std::endl;

            state->running.store(false);
            state->cv.notify_all();
            state->expert_cv.notify_all();
            break;
        }

        std::cerr << "Host rank=" << state->rank
                  << " ignored message type "
                  << header.type
                  << std::endl;
    }
}

static bool wait_for_dispatch_ack(
    HostState& state,
    const CombineKey& key
) {
    std::unique_lock<std::mutex> lock(state.state_mutex);

    state.cv.wait(lock, [&]() {
        return !state.running.load() ||
               state.dispatch_acks.find(key) != state.dispatch_acks.end();
    });

    if (!state.running.load()) {
        return false;
    }

    state.dispatch_acks.erase(key);
    return true;
}

enum class PullResult {
    READY,
    NOT_READY,
    STOPPED
};

static void prepare_pull_wait(
    HostState& state,
    const CombineKey& key
) {
    std::lock_guard<std::mutex> lock(state.state_mutex);

    auto& combine_state = state.combine_states[key];

    combine_state.not_ready = false;
}

static PullResult wait_for_pull_response(
    HostState& state,
    const CombineKey& key,
    MsgHeader& result_header,
    std::vector<char>& result_payload
) {
    std::unique_lock<std::mutex> lock(state.state_mutex);

    state.cv.wait(lock, [&]() {
        if (!state.running.load()) {
            return true;
        }

        auto it = state.combine_states.find(key);

        if (it == state.combine_states.end()) {
            return false;
        }

        return it->second.ready || it->second.not_ready;
    });

    if (!state.running.load()) {
        return PullResult::STOPPED;
    }

    auto it = state.combine_states.find(key);

    if (it == state.combine_states.end()) {
        return PullResult::STOPPED;
    }

    if (it->second.ready) {
        result_header = it->second.header;
        result_payload = it->second.payload;
        state.combine_states.erase(it);
        return PullResult::READY;
    }

    if (it->second.not_ready) {
        it->second.not_ready = false;
        return PullResult::NOT_READY;
    }

    return PullResult::STOPPED;
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: "
                  << argv[0]
                  << " <switch_ip> <switch_port> <rank> <num_tokens> <num_hosts> <topk>\n";
        return 1;
    }

    std::string switch_ip = argv[1];
    uint16_t switch_port = static_cast<uint16_t>(std::stoi(argv[2]));
    int rank = std::stoi(argv[3]);
    int num_tokens = std::stoi(argv[4]);
    int num_hosts = std::stoi(argv[5]);
    int topk = argc > 6 ? std::stoi(argv[6]) : std::min(2, num_hosts);

    if (rank < 0 || rank >= num_hosts) {
        std::cerr << "rank must be between 0 and num_hosts - 1"
                  << std::endl;
        return 2;
    }

    if (num_hosts <= 0 || num_hosts > 64) {
        std::cerr << "num_hosts must be between 1 and 64"
                  << std::endl;
        return 3;
    }

    if (topk <= 0) {
        topk = 1;
    }

    if (topk > num_hosts) {
        topk = num_hosts;
    }

    HostState state;
    state.rank = rank;

    if (!sr_connect(switch_ip, switch_port, state.conn)) {
        std::cerr << "Host rank=" << rank
                  << " failed to connect to switch "
                  << switch_ip
                  << ":"
                  << switch_port
                  << " by SoftRoCE"
                  << std::endl;
        return 4;
    }

    MsgHeader hello{};
    hello.magic = MAGIC;
    hello.version = PROTOCOL_VERSION;
    hello.type = MSG_HELLO;
    hello.header_len = sizeof(MsgHeader);
    hello.payload_len = 0;
    hello.run_id = 1;
    hello.src_rank = static_cast<uint32_t>(rank);
    hello.dst_rank = 0;
    hello.origin_rank = static_cast<uint32_t>(rank);
    hello.timestamp_ns = ns_timestamp();

    if (!send_to_switch(state, hello, nullptr)) {
        std::cerr << "Host rank=" << rank
                  << " failed to send HELLO"
                  << std::endl;

        sr_close(state.conn);
        return 5;
    }

    /*
     * HELLO ACK 由主线程同步读取。
     * 之后整个程序只能由 reader_loop 调用 sr_read()。
     */
    MsgHeader hello_reply{};
    std::vector<char> hello_payload;

    if (!sr_read(state.conn, hello_reply, hello_payload) ||
        hello_reply.type != MSG_ACK) {
        std::cerr << "Host rank=" << rank
                  << " did not receive HELLO ACK from switch"
                  << std::endl;

        sr_close(state.conn);
        return 6;
    }

    std::cout << "Host rank=" << rank
              << " connected and received HELLO ACK"
              << std::endl;

    std::thread reader_thread(reader_loop, &state);
    std::thread expert_thread(expert_worker_loop, &state);

    auto tokens = make_workload(
        hello.run_id,
        num_tokens,
        rank,
        num_hosts,
        topk
    );

    int received_combines = 0;

    for (const auto& token : tokens) {
        if (!state.running.load()) {
            break;
        }

        MsgHeader dispatch{};
        dispatch.magic = MAGIC;
        dispatch.version = PROTOCOL_VERSION;
        dispatch.type = MSG_DISPATCH_TOKEN;
        dispatch.header_len = sizeof(MsgHeader);
        dispatch.payload_len = token.payload_len;

        dispatch.run_id = token.run_id;
        dispatch.microbatch_id = token.microbatch_id;
        dispatch.token_id = token.token_id;

        dispatch.src_rank = token.src_rank;
        dispatch.dst_rank = 0;
        dispatch.origin_rank = token.origin_rank;
        dispatch.expert_bitmap = token.expert_bitmap;
        dispatch.global_idx = token.global_idx;
        dispatch.timestamp_ns = ns_timestamp();

        std::vector<char> payload(token.payload_len);

        for (uint32_t i = 0; i < token.payload_len; ++i) {
            payload[i] = static_cast<char>(token.token_id & 0xFF);
        }

        CombineKey key = make_key(dispatch);

        std::cout << "Host rank=" << rank
                  << " sending DISPATCH_TOKEN token="
                  << token.token_id
                  << " expert_bitmap="
                  << token.expert_bitmap
                  << std::endl;

        if (!send_to_switch(state, dispatch, payload.data())) {
            std::cerr << "Host rank=" << rank
                      << " failed to send DISPATCH_TOKEN token="
                      << token.token_id
                      << std::endl;

            state.running.store(false);
            state.cv.notify_all();
            break;
        }

        if (!wait_for_dispatch_ack(state, key)) {
            std::cerr << "Host rank=" << rank
                      << " failed to receive dispatch ACK token="
                      << token.token_id
                      << std::endl;

            state.running.store(false);
            state.cv.notify_all();
            break;
        }

        MsgHeader pull{};
        pull.magic = MAGIC;
        pull.version = PROTOCOL_VERSION;
        pull.type = MSG_COMBINE_PULL_REQ;
        pull.header_len = sizeof(MsgHeader);
        pull.payload_len = 0;

        pull.run_id = token.run_id;
        pull.microbatch_id = token.microbatch_id;
        pull.token_id = token.token_id;

        pull.src_rank = static_cast<uint32_t>(rank);
        pull.dst_rank = 0;
        pull.origin_rank = token.origin_rank;
        pull.global_idx = token.global_idx;
        pull.timestamp_ns = ns_timestamp();

        bool got_result = false;

        for (int attempt = 0; attempt < 100 && state.running.load(); ++attempt) {
            prepare_pull_wait(state, key);

            if (!send_to_switch(state, pull, nullptr)) {
                std::cerr << "Host rank=" << rank
                          << " failed to send COMBINE_PULL_REQ token="
                          << token.token_id
                          << std::endl;

                state.running.store(false);
                state.cv.notify_all();
                break;
            }

            MsgHeader result_header{};
            std::vector<char> result_payload;

            PullResult pull_result = wait_for_pull_response(
                state,
                key,
                result_header,
                result_payload
            );

            if (pull_result == PullResult::READY) {
                std::cout << "Host rank=" << rank
                          << " got COMBINE_RESULT token="
                          << result_header.token_id
                          << " experts="
                          << result_header.expert_bitmap
                          << " payload_len="
                          << result_header.payload_len
                          << std::endl;

                ++received_combines;
                got_result = true;
                break;
            }

            if (pull_result == PullResult::NOT_READY) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            std::cerr << "Host rank=" << rank
                      << " stopped while waiting COMBINE_RESULT token="
                      << token.token_id
                      << std::endl;

            break;
        }

        if (!got_result) {
            std::cerr << "Host rank=" << rank
                      << " timed out waiting COMBINE_RESULT token="
                      << token.token_id
                      << std::endl;

            state.running.store(false);
            state.cv.notify_all();
            break;
        }
    }

    if (state.running.load()) {
        MsgHeader finish{};
        finish.magic = MAGIC;
        finish.version = PROTOCOL_VERSION;
        finish.type = MSG_FINISH;
        finish.header_len = sizeof(MsgHeader);
        finish.payload_len = 0;
        finish.run_id = hello.run_id;
        finish.src_rank = static_cast<uint32_t>(rank);
        finish.dst_rank = 0;
        finish.origin_rank = static_cast<uint32_t>(rank);
        finish.timestamp_ns = ns_timestamp();

        std::cerr << "Host rank=" << rank
                      << " finished sending raw tokens"
                      << std::endl;

        send_to_switch(state, finish, nullptr);
    }

    if (reader_thread.joinable()) {
        reader_thread.join();
    }

    state.running.store(false);
    state.cv.notify_all();
    state.expert_cv.notify_all();

    if (expert_thread.joinable()) {
        expert_thread.join();
    }

    sr_close(state.conn);

    std::cout << "Host rank=" << rank
              << " completed, received "
              << received_combines
              << " combine results."
              << std::endl;

    return 0;
}