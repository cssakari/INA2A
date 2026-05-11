#include "transport_softroce.h"
#include "protocol.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct HostConn {
    int rank;
    SrConn* conn;
    std::mutex write_mutex;
};

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

struct QueueSlot {
    std::vector<char> data;

    uint64_t requested_bitmap;
    uint64_t pending_bitmap;
    uint64_t expected_bitmap;

    bool ready;

    uint64_t run_id;
    uint64_t microbatch_id;
    uint64_t token_id;
    uint32_t sender_rank;
    uint32_t origin_rank;
    uint64_t global_idx;
    uint32_t payload_len;
};

struct SwitchState {
    int num_hosts;
    int num_experts;

    std::map<int, std::shared_ptr<HostConn>> hosts;
    std::mutex hosts_mutex;

    std::map<CombineKey, QueueSlot> combine_queue;
    std::mutex queue_mutex;

    std::set<int> finished_hosts;
    std::mutex finish_mutex;

    std::atomic<bool> running{true};

    Logger logger{"switch.log"};
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

static std::string key_to_string(const CombineKey& key) {
    std::ostringstream oss;
    oss << "{run=" << key.run_id
        << ", mb=" << key.microbatch_id
        << ", token=" << key.token_id
        << ", origin=" << key.origin_rank
        << ", global=" << key.global_idx
        << "}";
    return oss.str();
}

static std::vector<uint32_t> decode_expert_ids(uint64_t bitmap, int num_experts) {
    std::vector<uint32_t> ids;

    for (int expert = 0; expert < num_experts && expert < 64; ++expert) {
        if ((bitmap >> expert) & 1ull) {
            ids.push_back(static_cast<uint32_t>(expert));
        }
    }

    if (ids.empty() && num_experts > 0) {
        ids.push_back(0);
    }

    return ids;
}

static std::shared_ptr<HostConn> find_host(SwitchState& state, int rank) {
    std::lock_guard<std::mutex> lock(state.hosts_mutex);

    auto it = state.hosts.find(rank);

    if (it == state.hosts.end()) {
        return nullptr;
    }

    return it->second;
}

static std::vector<std::shared_ptr<HostConn>> snapshot_hosts(SwitchState& state) {
    std::vector<std::shared_ptr<HostConn>> result;

    std::lock_guard<std::mutex> lock(state.hosts_mutex);

    for (auto& kv : state.hosts) {
        result.push_back(kv.second);
    }

    return result;
}

static bool send_to_host(
    const std::shared_ptr<HostConn>& host,
    const MsgHeader& header,
    const void* payload
) {
    if (!host || !host->conn) {
        return false;
    }

    std::lock_guard<std::mutex> lock(host->write_mutex);

    return sr_write(host->conn, header, payload);
}

static void send_ack_to_host(
    const std::shared_ptr<HostConn>& host,
    const MsgHeader& src
) {
    MsgHeader ack{};
    ack.magic = MAGIC;
    ack.version = PROTOCOL_VERSION;
    ack.type = MSG_ACK;
    ack.header_len = sizeof(MsgHeader);
    ack.payload_len = 0;
    ack.run_id = src.run_id;
    ack.microbatch_id = src.microbatch_id;
    ack.token_id = src.token_id;
    ack.src_rank = 0;
    ack.dst_rank = src.src_rank;
    ack.origin_rank = src.origin_rank;
    ack.expert_bitmap = src.expert_bitmap;
    ack.global_idx = src.global_idx;
    ack.timestamp_ns = ns_timestamp();

    if (!send_to_host(host, ack, nullptr)) {
        std::cerr << "Switch failed to send ACK to host rank "
                  << host->rank
                  << std::endl;
    }
}

static void broadcast_finish(SwitchState& state, uint64_t run_id) {
    auto hosts = snapshot_hosts(state);

    for (auto& host : hosts) {
        MsgHeader finish{};
        finish.magic = MAGIC;
        finish.version = PROTOCOL_VERSION;
        finish.type = MSG_FINISH;
        finish.header_len = sizeof(MsgHeader);
        finish.payload_len = 0;
        finish.run_id = run_id;
        finish.src_rank = 0;
        finish.dst_rank = static_cast<uint32_t>(host->rank);
        finish.timestamp_ns = ns_timestamp();

        send_to_host(host, finish, nullptr);
    }
}

static bool mark_host_finished(
    SwitchState& state,
    int rank
) {
    std::lock_guard<std::mutex> lock(state.finish_mutex);

    state.finished_hosts.insert(rank);

    state.logger.logf(Logger::INFO, "Switch marked host rank %d as finished. finished=%u/%u", rank, static_cast<uint32_t>(state.finished_hosts.size()), static_cast<uint32_t>(state.num_hosts));

    return static_cast<int>(state.finished_hosts.size()) >= state.num_hosts;
}

static void close_all_host_connections(SwitchState& state) {
    auto hosts = snapshot_hosts(state);

    for (auto& host : hosts) {
        if (host && host->conn) {
            sr_close(host->conn);
            host->conn = nullptr;
        }
    }
}

static void handle_expert_result(
    SwitchState& state,
    const std::shared_ptr<HostConn>& host,
    const MsgHeader& header,
    const std::vector<char>& payload
) {
    CombineKey key = make_key(header);

    std::lock_guard<std::mutex> lock(state.queue_mutex);

    auto it_slot = state.combine_queue.find(key);

    if (it_slot == state.combine_queue.end()) {

        state.logger.logf(Logger::ERROR, "Switch received EXPERT_RESULT for unknown token %s from rank %d", key_to_string(key).c_str(), host->rank);
        return;
    }

    QueueSlot& slot = it_slot->second;

    if (payload.size() != slot.payload_len) {

        state.logger.logf(Logger::ERROR, "Switch received size mismatch from rank %d, expected %u, got %zu", host->rank, slot.payload_len, payload.size());
        return;
    }

    uint32_t expert_rank = header.src_rank;

    if (expert_rank >= 64) {
        state.logger.logf(Logger::ERROR, "Switch received invalid expert rank %d", expert_rank);
        return;
    }

    uint64_t expert_bit = 1ull << expert_rank;

    if ((slot.pending_bitmap & expert_bit) == 0) {
        state.logger.logf(Logger::ERROR, "Switch received duplicate or unexpected EXPERT_RESULT from rank %d for token %s", expert_rank, key_to_string(key).c_str());
        return;
    }

    for (size_t i = 0; i < payload.size(); ++i) {
        slot.data[i] = static_cast<char>(
            (
                static_cast<uint8_t>(slot.data[i]) +
                static_cast<uint8_t>(payload[i])
            ) & 0xFF
        );
    }

    slot.pending_bitmap &= ~expert_bit;

    state.logger.logf(Logger::INFO, "Switch aggregated EXPERT_RESULT token=%u from expert rank=%d, original rank=%d, pending_bitmap=%lu", header.token_id, expert_rank, slot.origin_rank, slot.pending_bitmap);

    if (slot.pending_bitmap == 0) {
        slot.ready = true;

        state.logger.logf(Logger::INFO, "Switch token %s fully aggregated.", key_to_string(key).c_str());
    }
}

static void handle_dispatch_token(
    SwitchState& state,
    const std::shared_ptr<HostConn>& sender,
    const MsgHeader& header,
    const std::vector<char>& payload
) {
    if (payload.size() != header.payload_len) {

        state.logger.logf(Logger::ERROR, "Switch received DISPATCH_TOKEN payload size mismatch from rank %d, header payload_len=%u, actual=%zu",
            sender->rank,
            header.payload_len,
            payload.size()
        );
        return;
    }

    CombineKey key = make_key(header);

    auto selected_experts = decode_expert_ids(header.expert_bitmap, state.num_experts);

    state.logger.logf(Logger::INFO, "Switch received DISPATCH_TOKEN token=%u from sender rank=%d, selected_experts=%zu",
        header.token_id,
        sender->rank,
        selected_experts.size()
    );

    std::vector<std::shared_ptr<HostConn>> targets;
    uint64_t expected_bitmap = 0;

    for (uint32_t expert_rank : selected_experts) {
        auto target = find_host(state, static_cast<int>(expert_rank));

        if (!target) {
            state.logger.logf(Logger::ERROR, "Switch has no connected host for expert rank %d, this expert will be skipped.", expert_rank);
            continue;
        }

        targets.push_back(target);
        expected_bitmap |= (1ull << expert_rank);
    }

    {
        std::lock_guard<std::mutex> lock(state.queue_mutex);

        if (state.combine_queue.count(key)) {
            state.logger.logf(Logger::INFO, "Switch overwriting existing combine slot for token %s", key_to_string(key).c_str());
        }

        QueueSlot slot{};
        slot.data.assign(header.payload_len, 0);
        slot.requested_bitmap = header.expert_bitmap;
        slot.pending_bitmap = expected_bitmap;
        slot.expected_bitmap = expected_bitmap;
        slot.ready = expected_bitmap == 0;

        slot.run_id = header.run_id;
        slot.microbatch_id = header.microbatch_id;
        slot.token_id = header.token_id;
        slot.sender_rank = header.src_rank;
        slot.origin_rank = header.origin_rank;
        slot.global_idx = header.global_idx;
        slot.payload_len = header.payload_len;

        state.combine_queue[key] = std::move(slot);
    }

    for (auto& target : targets) {
        MsgHeader input_header{};
        input_header.magic = MAGIC;
        input_header.version = PROTOCOL_VERSION;
        input_header.type = MSG_EXPERT_INPUT;
        input_header.header_len = sizeof(MsgHeader);
        input_header.payload_len = header.payload_len;
        input_header.run_id = header.run_id;
        input_header.microbatch_id = header.microbatch_id;
        input_header.token_id = header.token_id;

        input_header.src_rank = 0;
        input_header.dst_rank = static_cast<uint32_t>(target->rank);
        input_header.origin_rank = header.origin_rank;

        input_header.expert_bitmap = header.expert_bitmap;
        input_header.global_idx = header.global_idx;
        input_header.timestamp_ns = ns_timestamp();

        if (!send_to_host(target, input_header, payload.data())) {
            state.logger.logf(Logger::ERROR, "Switch failed to send EXPERT_INPUT token=%u from rank=%d to expert rank=%d", header.token_id, header.src_rank, target->rank);

            std::lock_guard<std::mutex> lock(state.queue_mutex);

            auto it_slot = state.combine_queue.find(key);

            if (it_slot != state.combine_queue.end()) {
                it_slot->second.pending_bitmap &= ~(1ull << target->rank);

                if (it_slot->second.pending_bitmap == 0) {
                    it_slot->second.ready = true;
                }
            }
        } else {
            state.logger.logf(Logger::INFO, "Switch dispatched token=%u from rank=%d to expert rank=%d", header.token_id, header.src_rank, target->rank);
        }
    }
    
    send_ack_to_host(sender, header);
}

static void handle_combine_pull_req(
    SwitchState& state,
    const std::shared_ptr<HostConn>& sender,
    const MsgHeader& header
) {
    CombineKey key = make_key(header);

    std::vector<char> result_data;
    uint32_t result_payload_len = 0;
    uint64_t result_expert_bitmap = 0;
    bool ready = false;
    bool exists = false;

    {
        std::lock_guard<std::mutex> lock(state.queue_mutex);

        auto it_slot = state.combine_queue.find(key);

        if (it_slot != state.combine_queue.end()) {
            exists = true;

            if (it_slot->second.ready) {
                result_data = it_slot->second.data;
                result_payload_len = it_slot->second.payload_len;
                result_expert_bitmap = it_slot->second.expected_bitmap;
                ready = true;

                state.combine_queue.erase(it_slot);
            }
        }
    }

    if (ready) {
        MsgHeader response{};
        response.magic = MAGIC;
        response.version = PROTOCOL_VERSION;
        response.type = MSG_COMBINE_RESULT;
        response.header_len = sizeof(MsgHeader);
        response.payload_len = result_payload_len;
        response.run_id = header.run_id;
        response.microbatch_id = header.microbatch_id;
        response.token_id = header.token_id;
        response.src_rank = 0;
        response.dst_rank = header.src_rank;
        response.origin_rank = header.origin_rank;
        response.expert_bitmap = result_expert_bitmap;
        response.global_idx = header.global_idx;
        response.timestamp_ns = ns_timestamp();

        if (!send_to_host(sender, response, result_data.data())) {
            state.logger.logf(Logger::ERROR, "Switch failed to send COMBINE_RESULT to rank %d", sender->rank);
        }

        return;
    }

    MsgHeader not_ready{};
    not_ready.magic = MAGIC;
    not_ready.version = PROTOCOL_VERSION;
    not_ready.type = MSG_NOT_READY;
    not_ready.header_len = sizeof(MsgHeader);
    not_ready.payload_len = 0;
    not_ready.run_id = header.run_id;
    not_ready.microbatch_id = header.microbatch_id;
    not_ready.token_id = header.token_id;
    not_ready.src_rank = 0;
    not_ready.dst_rank = header.src_rank;
    not_ready.origin_rank = header.origin_rank;
    not_ready.global_idx = header.global_idx;
    not_ready.timestamp_ns = ns_timestamp();

    if (!exists) {
        state.logger.logf(Logger::ERROR, "Switch got COMBINE_PULL_REQ but slot does not exist for token %s", key_to_string(key).c_str());
    }

    if (!send_to_host(sender, not_ready, nullptr)) {
        state.logger.logf(Logger::ERROR, "Switch failed to send MSG_NOT_READY to rank %d", sender->rank);
    }
}

static void host_reader_loop(
    SwitchState* state,
    std::shared_ptr<HostConn> host
) {
    while (state->running.load()) {
        MsgHeader header{};
        std::vector<char> payload;

        if (!host->conn) {
            break;
        }

        if (!sr_read(host->conn, header, payload)) {
            if (state->running.load()) {
                state->logger.logf(Logger::ERROR, "Switch receive failed from host rank %d", host->rank);
            }
            break;
        }

        if (header.type == MSG_DISPATCH_TOKEN) {
            handle_dispatch_token(*state, host, header, payload);
            continue;
        }

        if (header.type == MSG_EXPERT_RESULT) {
            handle_expert_result(*state, host, header, payload);
            continue;
        }

        if (header.type == MSG_COMBINE_PULL_REQ) {
            handle_combine_pull_req(*state, host, header);
            continue;
        }

        if (header.type == MSG_ACK) {
            state->logger.logf(Logger::INFO, "Switch received ACK from host rank %d", host->rank);
            continue;
        }

        if (header.type == MSG_FINISH) {

            state->logger.logf(Logger::INFO, "Switch received FINISH from host rank %d", host->rank);

            bool all_finished = mark_host_finished(*state, host->rank);

            if (all_finished) {
                state->logger.logf(Logger::INFO, "Switch received FINISH from all hosts, broadcasting FINISH.");

                broadcast_finish(*state, header.run_id);
                state->running.store(false);
                close_all_host_connections(*state);

                break;
            }

            continue;
        }

        state->logger.logf(Logger::ERROR, "Switch ignored message type %d from host rank %d", header.type, host->rank);
    }

    state->logger.logf(Logger::INFO, "Switch reader loop exited for host rank %d", host->rank);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: "
                  << argv[0]
                  << " <listen_port> <num_hosts>\n";
        return 1;
    }

    uint16_t listen_port = static_cast<uint16_t>(std::stoi(argv[1]));
    int num_hosts = std::stoi(argv[2]);

    if (num_hosts <= 0 || num_hosts > 64) {
        std::cerr << "num_hosts must be between 1 and 64" << std::endl;
        return 2;
    }

    int num_experts = num_hosts;

    SrListener* listener = nullptr;

    if (!sr_listen(listen_port, listener)) {
        std::cerr << "Switch cannot listen on port "
                  << listen_port
                  << " by SoftRoCE"
                  << std::endl;
        return 3;
    }

    std::cout << "Switch listening on port "
              << listen_port
              << ", num_hosts="
              << num_hosts
              << ", num_experts="
              << num_experts
              << std::endl;

    SwitchState state;
    state.num_hosts = num_hosts;
    state.num_experts = num_experts;

    std::vector<std::shared_ptr<HostConn>> accepted_hosts;

    for (int i = 0; i < num_hosts; ++i) {
        SrConn* conn = nullptr;

        std::cout << "Switch waiting for host connection "
                  << (i + 1)
                  << "/"
                  << num_hosts
                  << std::endl;

        if (!sr_accept(listener, conn)) {
            std::cerr << "Switch failed to accept host connection"
                      << std::endl;

            sr_close_listener(listener);

            for (auto& host : accepted_hosts) {
                sr_close(host->conn);
            }

            return 4;
        }

        MsgHeader hello{};
        std::vector<char> hello_payload;

        if (!sr_read(conn, hello, hello_payload) || hello.type != MSG_HELLO) {
            std::cerr << "Switch accepted a connection but did not receive MSG_HELLO"
                      << std::endl;

            sr_close(conn);
            --i;
            continue;
        }

        int rank = static_cast<int>(hello.src_rank);

        if (rank < 0 || rank >= num_hosts) {
            std::cerr << "Switch received invalid host rank "
                      << rank
                      << std::endl;

            sr_close(conn);
            --i;
            continue;
        }

        auto host = std::make_shared<HostConn>();
        host->rank = rank;
        host->conn = conn;

        {
            std::lock_guard<std::mutex> lock(state.hosts_mutex);

            if (state.hosts.count(rank)) {
                std::cerr << "Switch received duplicate host rank "
                          << rank
                          << std::endl;

                sr_close(conn);
                --i;
                continue;
            }

            state.hosts[rank] = host;
        }

        accepted_hosts.push_back(host);

        std::cout << "Switch accepted host rank "
                  << rank
                  << std::endl;

        MsgHeader ack{};
        ack.magic = MAGIC;
        ack.version = PROTOCOL_VERSION;
        ack.type = MSG_ACK;
        ack.header_len = sizeof(MsgHeader);
        ack.payload_len = 0;
        ack.run_id = hello.run_id;
        ack.src_rank = 0;
        ack.dst_rank = hello.src_rank;
        ack.timestamp_ns = ns_timestamp();

        if (!send_to_host(host, ack, nullptr)) {
            std::cerr << "Switch failed to send HELLO ACK to host rank "
                      << rank
                      << std::endl;

            sr_close(conn);
            return 5;
        }
    }

    std::cout << "Switch accepted all hosts. Starting dispatch/combine loops."
              << std::endl;

    std::vector<std::thread> threads;

    for (auto& host : accepted_hosts) {
        threads.emplace_back(host_reader_loop, &state, host);
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    state.running.store(false);

    for (auto& host : accepted_hosts) {
        if (host && host->conn) {
            sr_close(host->conn);
            host->conn = nullptr;
        }
    }

    sr_close_listener(listener);

    std::cout << "Switch closed normally." << std::endl;

    return 0;
}