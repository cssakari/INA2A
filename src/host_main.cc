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
#include <condition_variable>
#include <optional>

template <typename T>
class BlockingQueue {
public:
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (closed_) {
                return;
            }
            q_.push_back(std::move(item));
        }
        cv_.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mu_);

        cv_.wait(lock, [&]() {
            return closed_ || !q_.empty();
        });

        if (q_.empty()) {
            return false;
        }

        item = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    bool try_pop_for(T& item, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);

        cv_.wait_for(lock, timeout, [&]() {
            return closed_ || !q_.empty();
        });

        if (q_.empty()) {
            return false;
        }

        item = std::move(q_.front());
        q_.pop_front();
        return true;
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<T> q_;
    bool closed_ = false;
};

struct OutMsg {
    MsgHeader header{};
    std::vector<char> payload;
};

enum class ComputeTaskType {
    LOCAL_ATTENTION,
    EXPERT_MLP
};

struct ComputeTask {
    ComputeTaskType type;
    int token_index = -1;
    MsgHeader header{};
    std::vector<char> payload;
};

struct TokenRuntime {
    bool dispatch_sent = false;
    bool dispatch_acked = false;
    bool combine_done = false;
    bool combine_requested_once = false;

    int pull_attempts = 0;

    uint64_t token_id = 0;
    uint64_t microbatch_id = 0;
    uint64_t global_idx = 0;
    uint32_t origin_rank = 0;
    uint64_t expert_bitmap = 0;
    uint32_t payload_len = 0;

    std::vector<char> payload;
    std::vector<char> combine_result;
};

struct TokenWindowState {
    bool admitted = false;          // 已经进入滑动窗口
    bool attention_queued = false;  // 已经放入compute_queue等待attention
    bool attention_done = false;    // attention已经完成

    bool dispatch_enqueued = false; // DISPATCH_TOKEN已经放入send_queue
    bool dispatch_acked = false;    // 收到Switch的ACK

    bool pull_inflight = false;     // 当前是否已有COMBINE_PULL_REQ在等待回复
    int pull_attempts = 0;

    bool combine_done = false;      // 收到COMBINE_RESULT，生命周期结束

    std::chrono::steady_clock::time_point next_pull_time =
        std::chrono::steady_clock::now();

    std::vector<char> combine_payload;
};

struct HostState {
    int rank = -1;
    int num_hosts = 0;
    int topk = 1;
    int num_tokens = 0;
    int window_size = 1;

    uint64_t run_id = 1;

    SrConn* conn = nullptr;

    std::atomic<bool> running{true};

    BlockingQueue<OutMsg> send_queue;
    BlockingQueue<ComputeTask> compute_queue;

    std::vector<TokenDesc> workload;
    std::vector<TokenWindowState> token_states;

    std::mutex write_mutex;
    std::mutex state_mutex;
    std::condition_variable cv;

    int window_base = 0;
    int next_to_admit = 0;

    int received_combines = 0;
    bool finish_enqueued = false;

    /*
     * token index -> runtime
     */
    std::map<int, TokenRuntime> tokens;
};

static void enqueue_send(
    HostState& state,
    const MsgHeader& header,
    const void* payload
) {
    OutMsg msg;
    msg.header = header;

    if (payload != nullptr && header.payload_len > 0) {
        const char* p = static_cast<const char*>(payload);
        msg.payload.assign(p, p + header.payload_len);
    }

    state.send_queue.push(std::move(msg));
}

static void enqueue_combine_pull(
    HostState& state,
    int token_index
) {
    if (token_index < 0 ||
        token_index >= static_cast<int>(state.workload.size())) {
        return;
    }

    const auto& token = state.workload[token_index];

    MsgHeader pull{};
    pull.magic = MAGIC;
    pull.version = PROTOCOL_VERSION;
    pull.type = MSG_COMBINE_PULL_REQ;
    pull.header_len = sizeof(MsgHeader);
    pull.payload_len = 0;

    pull.run_id = token.run_id;
    pull.microbatch_id = token.microbatch_id;
    pull.token_id = token.token_id;

    pull.src_rank = static_cast<uint32_t>(state.rank);
    pull.dst_rank = 0;
    pull.origin_rank = token.origin_rank;
    pull.expert_bitmap = token.expert_bitmap;

    /*
     * 必须和DISPATCH_TOKEN一致，保持token_index定位。
     */
    pull.global_idx = static_cast<uint64_t>(token_index);
    pull.timestamp_ns = ns_timestamp();

    enqueue_send(state, pull, nullptr);
}

static void process_expert_mlp(
    HostState& state,
    const MsgHeader& input_header,
    const std::vector<char>& payload
) {
    std::cout << "Host rank=" << state.rank
              << " compute EXPERT_INPUT token="
              << input_header.token_id
              << " origin_rank="
              << input_header.origin_rank
              << std::endl;

    std::vector<char> result(payload.size());

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

    enqueue_send(state, response, result.data());

    std::cout << "Host rank=" << state.rank
              << " enqueue EXPERT_RESULT token="
              << input_header.token_id
              << " origin_rank="
              << input_header.origin_rank
              << std::endl;
}

static void try_admit_raw_tokens_to_compute_queue(HostState& state) {
    std::vector<int> admitted_indices;

    {
        std::lock_guard<std::mutex> lock(state.state_mutex);

        while (state.next_to_admit < state.num_tokens &&
               state.next_to_admit < state.window_base + state.window_size) {
            int idx = state.next_to_admit;

            auto& st = state.token_states[idx];

            st.admitted = true;
            st.attention_queued = true;

            admitted_indices.push_back(idx);

            state.next_to_admit++;
        }
    }

    for (int idx : admitted_indices) {
        ComputeTask task;
        task.type = ComputeTaskType::LOCAL_ATTENTION;
        task.token_index = idx;

        state.compute_queue.push(std::move(task));

        std::cout << "Host rank=" << state.rank
                  << " admit raw token to compute_queue, token_index="
                  << idx
                  << std::endl;
    }
}

static void process_local_attention(
    HostState& state,
    int token_index
) {
    if (token_index < 0 ||
        token_index >= static_cast<int>(state.workload.size())) {
        return;
    }

    const auto& token = state.workload[token_index];

    /*
     * 模拟attention层输出。
     * 注意：
     *     payload不是make_workload阶段一次性生成的，
     *     而是在窗口允许后，由compute_loop动态生成。
     */
    std::vector<char> payload(token.payload_len);

    for (uint32_t i = 0; i < token.payload_len; ++i) {
        payload[i] = static_cast<char>(
            (token.token_id + i) & 0xFF
        );
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

    /*
     * 这里建议global_idx直接等于token_index。
     * 这样read_loop收到ACK / COMBINE_RESULT时，可以直接定位token_states[token_index]。
     */
    dispatch.global_idx = static_cast<uint64_t>(token_index);
    dispatch.timestamp_ns = ns_timestamp();

    {
        std::lock_guard<std::mutex> lock(state.state_mutex);

        auto& st = state.token_states[token_index];

        st.attention_done = true;
        st.dispatch_enqueued = true;
    }

    enqueue_send(state, dispatch, payload.data());

    std::cout << "Host rank=" << state.rank
              << " attention done, enqueue DISPATCH_TOKEN token="
              << token.token_id
              << " token_index="
              << token_index
              << " expert_bitmap="
              << token.expert_bitmap
              << std::endl;

    state.cv.notify_all();
}

//unsafe to call concurrently with send_loop
static bool send_to_switch(
    HostState& state,
    const MsgHeader& header,
    const void* payload
) {
    return sr_write(state.conn, header, payload);
}

static void compute_loop(HostState* state) {
    /*
     * 初始只允许window_size个raw token进入attention。
     * window_size=1时，只会加入第一个raw token。
     */
    try_admit_raw_tokens_to_compute_queue(*state);

    while (state->running.load()) {
        /*
         * 1. 处理计算任务：
         *    - LOCAL_ATTENTION：本host自己的token生成/attention
         *    - EXPERT_MLP：别人dispatch过来的expert计算
         */
        ComputeTask task;

        if (state->compute_queue.try_pop_for(
                task,
                std::chrono::milliseconds(1)
            )) {
            if (task.type == ComputeTaskType::LOCAL_ATTENTION) {
                process_local_attention(*state, task.token_index);
            } else if (task.type == ComputeTaskType::EXPERT_MLP) {
                process_expert_mlp(*state, task.header, task.payload);
            }

            continue;
        }

        /*
         * 2. 对已经dispatch ACK，但尚未combine完成的token，周期性发pull。
         */
        auto now = std::chrono::steady_clock::now();
        std::vector<int> need_pull;

        {
            std::lock_guard<std::mutex> lock(state->state_mutex);

            for (int i = state->window_base;
                 i < state->next_to_admit;
                 ++i) {
                if (i < 0 ||
                    i >= static_cast<int>(state->token_states.size())) {
                    continue;
                }

                auto& st = state->token_states[i];

                if (!st.admitted) {
                    continue;
                }

                if (!st.attention_done) {
                    continue;
                }

                if (!st.dispatch_acked) {
                    continue;
                }

                if (st.combine_done) {
                    continue;
                }

                if (st.pull_inflight) {
                    continue;
                }

                if (now < st.next_pull_time) {
                    continue;
                }

                st.pull_inflight = true;
                need_pull.push_back(i);
            }
        }

        for (int idx : need_pull) {
            enqueue_combine_pull(*state, idx);
        }

        /*
         * 3. 滑动窗口：
         *    只有window_base对应token收到COMBINE_RESULT，
         *    才能释放窗口槽位。
         */
        bool window_moved = false;

        {
            std::lock_guard<std::mutex> lock(state->state_mutex);

            while (state->window_base < state->num_tokens) {
                int idx = state->window_base;

                if (idx < 0 ||
                    idx >= static_cast<int>(state->token_states.size())) {
                    break;
                }

                if (!state->token_states[idx].combine_done) {
                    break;
                }

                std::cout << "Host rank=" << state->rank
                          << " sliding window releases token_index="
                          << idx
                          << std::endl;

                state->window_base++;
                state->received_combines++;
                window_moved = true;
            }
        }

        /*
         * 4. 只要窗口释放了槽位，新的raw token才进入compute_queue。
         */
        if (window_moved) {
            try_admit_raw_tokens_to_compute_queue(*state);
        }

        /*
         * 5. 本host作为sender完成全部token后，发送FINISH。
         *    注意不要立刻running=false，因为还要继续作为expert处理别人发来的token。
         */
        {
            std::lock_guard<std::mutex> lock(state->state_mutex);

            if (!state->finish_enqueued &&
                state->received_combines >= state->num_tokens) {
                MsgHeader finish{};
                finish.magic = MAGIC;
                finish.version = PROTOCOL_VERSION;
                finish.type = MSG_FINISH;
                finish.header_len = sizeof(MsgHeader);
                finish.payload_len = 0;

                finish.run_id = state->workload.empty()
                                    ? 0
                                    : state->workload[0].run_id;
                finish.src_rank = static_cast<uint32_t>(state->rank);
                finish.dst_rank = 0;
                finish.origin_rank = static_cast<uint32_t>(state->rank);
                finish.timestamp_ns = ns_timestamp();

                enqueue_send(*state, finish, nullptr);

                state->finish_enqueued = true;

                std::cout << "Host rank=" << state->rank
                          << " local sender finished all tokens"
                          << std::endl;
            }
        }
    }

    std::cout << "Host rank=" << state->rank
              << " compute_loop exited"
              << std::endl;
}

static void send_loop(HostState* state) {
    while (state->running.load()) {
        OutMsg msg;

        if (!state->send_queue.pop(msg)) {
            break;
        }

        const void* payload_ptr = nullptr;

        if (!msg.payload.empty()) {
            payload_ptr = msg.payload.data();
        }

        if (!send_to_switch(*state, msg.header, payload_ptr)) {
            std::cerr << "Host rank=" << state->rank
                      << " send_loop failed, msg_type="
                      << msg.header.type
                      << " token="
                      << msg.header.token_id
                      << std::endl;

            state->running.store(false);
            state->cv.notify_all();
            state->send_queue.close();
            state->compute_queue.close();
            break;
        }
    }

    std::cout << "Host rank=" << state->rank
              << " send_loop exited"
              << std::endl;
}

static void read_loop(HostState* state) {
    while (state->running.load()) {
        MsgHeader header{};
        std::vector<char> payload;

        if (!sr_read(state->conn, header, payload)) {
            std::cerr << "Host rank=" << state->rank
                      << " read_loop failed"
                      << std::endl;

            state->running.store(false);
            state->cv.notify_all();
            state->send_queue.close();
            state->compute_queue.close();
            break;
        }

        int idx = static_cast<int>(header.global_idx);

        if (header.type == MSG_ACK) {
            {
                std::lock_guard<std::mutex> lock(state->state_mutex);

                if (idx >= 0 &&
                    idx < static_cast<int>(state->token_states.size())) {
                    state->token_states[idx].dispatch_acked = true;
                }
            }

            state->cv.notify_all();
            continue;
        }

        if (header.type == MSG_EXPERT_INPUT) {
            ComputeTask task;
            task.type = ComputeTaskType::EXPERT_MLP;
            task.header = header;
            task.payload = std::move(payload);

            state->compute_queue.push(std::move(task));
            continue;
        }

        if (header.type == MSG_NOT_READY) {
            {
                std::lock_guard<std::mutex> lock(state->state_mutex);

                if (idx >= 0 &&
                    idx < static_cast<int>(state->token_states.size())) {
                    auto& st = state->token_states[idx];

                    st.pull_inflight = false;
                    st.pull_attempts++;

                    st.next_pull_time =
                        std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(10);
                }
            }

            state->cv.notify_all();
            continue;
        }

        if (header.type == MSG_COMBINE_RESULT) {
            {
                std::lock_guard<std::mutex> lock(state->state_mutex);

                if (idx >= 0 &&
                    idx < static_cast<int>(state->token_states.size())) {
                    auto& st = state->token_states[idx];

                    st.combine_done = true;
                    st.pull_inflight = false;
                    st.combine_payload = std::move(payload);

                    std::cout << "Host rank=" << state->rank
                              << " received COMBINE_RESULT token="
                              << header.token_id
                              << " token_index="
                              << idx
                              << std::endl;
                }
            }

            state->cv.notify_all();
            continue;
        }

        if (header.type == MSG_FINISH) {
            std::cout << "Host rank=" << state->rank
                      << " received FINISH"
                      << std::endl;

            state->running.store(false);
            state->cv.notify_all();
            state->send_queue.close();
            state->compute_queue.close();
            break;
        }

        std::cerr << "Host rank=" << state->rank
                  << " ignored msg_type="
                  << header.type
                  << std::endl;
    }
}

enum class PullResult {
    READY,
    NOT_READY,
    STOPPED
};

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: "
                  << argv[0]
                  << " <switch_ip> <switch_port> <rank> <num_tokens> <num_hosts> <topk> <window_size>\n";
        return 1;
    }

    std::string switch_ip = argv[1];
    uint16_t switch_port = static_cast<uint16_t>(std::stoi(argv[2]));
    int rank = std::stoi(argv[3]);
    int num_tokens = std::stoi(argv[4]);
    int num_hosts = std::stoi(argv[5]);
    int topk = argc > 6 ? std::stoi(argv[6]) : std::min(2, num_hosts);
    int window_size = argc > 7 ? std::stoi(argv[7]) : 1;

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

    auto tokens = make_workload(
    hello.run_id,
    num_tokens,
    rank,
    num_hosts,
    topk
);

{
    std::lock_guard<std::mutex> lock(state.state_mutex);

    state.rank = rank;

    state.workload = std::move(tokens);
    state.num_tokens = static_cast<int>(state.workload.size());

    state.window_size = window_size;

    state.window_base = 0;
    state.next_to_admit = 0;
    state.received_combines = 0;
    state.finish_enqueued = false;

    state.token_states.clear();
    state.token_states.resize(state.num_tokens);
}

/*
 * 从这里开始，main不再直接发送token。
 * raw token必须先被滑动窗口放入compute_queue，
 * attention完成后才会进入send_queue。
 */
std::thread sender_thread(send_loop, &state);
std::thread reader_thread(read_loop, &state);
std::thread compute_thread(compute_loop, &state);

if (compute_thread.joinable()) {
    compute_thread.join();
}

state.running.store(false);
state.cv.notify_all();
state.send_queue.close();
state.compute_queue.close();

if (reader_thread.joinable()) {
    reader_thread.join();
}

if (sender_thread.joinable()) {
    sender_thread.join();
}

std::cout << "Host rank=" << rank
          << " exited, received_combines="
          << state.received_combines
          << "/"
          << state.num_tokens
          << std::endl;

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

    if (compute_thread.joinable()) {
        compute_thread.join();
    }

    state.running.store(false);
    state.cv.notify_all();
    state.compute_queue.close();
    state.send_queue.close();

    if (reader_thread.joinable()) {
        reader_thread.join();
    }

    if (sender_thread.joinable()) {
        sender_thread.join();
    }

    sr_close(state.conn);

    std::cout << "Host rank=" << rank
              << " completed, received "
              << state.received_combines
              << " combine results."
              << std::endl;

    return 0;
}