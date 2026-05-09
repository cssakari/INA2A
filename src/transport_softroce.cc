#include "transport_softroce.h"

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <rdma/rdma_cma.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>
#include <unistd.h>
#include <chrono>

struct SrListener {
    rdma_cm_id* id = nullptr;
};

struct SrConn {
    rdma_cm_id* id = nullptr;

    ibv_pd* pd = nullptr;
    ibv_cq* send_cq = nullptr;
    ibv_cq* recv_cq = nullptr;

    ibv_mr* send_mr = nullptr;
    ibv_mr* recv_mr = nullptr;

    char* send_buf = nullptr;
    char* recv_buf = nullptr;
    size_t buf_size = SR_MAX_MSG_SIZE;

    bool recv_posted = false;
};

static uint64_t host_to_network_u64(uint64_t value) {
    static const uint16_t one = 1;
    if (*reinterpret_cast<const char*>(&one) == 1) {
        uint64_t lo = htonl(static_cast<uint32_t>(value & 0xFFFFFFFFULL));
        uint64_t hi = htonl(static_cast<uint32_t>(value >> 32));
        return (lo << 32) | hi;
    }
    return value;
}

static uint64_t network_to_host_u64(uint64_t value) {
    static const uint16_t one = 1;
    if (*reinterpret_cast<const char*>(&one) == 1) {
        uint64_t lo = ntohl(static_cast<uint32_t>(value & 0xFFFFFFFFULL));
        uint64_t hi = ntohl(static_cast<uint32_t>(value >> 32));
        return (lo << 32) | hi;
    }
    return value;
}

static MsgHeader header_to_wire(MsgHeader h) {
    h.magic = htonl(h.magic);
    h.version = htons(h.version);
    h.type = htons(h.type);
    h.header_len = htonl(h.header_len);
    h.payload_len = htonl(h.payload_len);

    h.run_id = host_to_network_u64(h.run_id);
    h.microbatch_id = host_to_network_u64(h.microbatch_id);
    h.token_id = host_to_network_u64(h.token_id);

    h.src_rank = htonl(h.src_rank);
    h.dst_rank = htonl(h.dst_rank);
    h.origin_rank = htonl(h.origin_rank);

    h.expert_bitmap = host_to_network_u64(h.expert_bitmap);
    h.global_idx = host_to_network_u64(h.global_idx);
    h.timestamp_ns = host_to_network_u64(h.timestamp_ns);

    return h;
}

static MsgHeader header_from_wire(MsgHeader h) {
    h.magic = ntohl(h.magic);
    h.version = ntohs(h.version);
    h.type = ntohs(h.type);
    h.header_len = ntohl(h.header_len);
    h.payload_len = ntohl(h.payload_len);

    h.run_id = network_to_host_u64(h.run_id);
    h.microbatch_id = network_to_host_u64(h.microbatch_id);
    h.token_id = network_to_host_u64(h.token_id);

    h.src_rank = ntohl(h.src_rank);
    h.dst_rank = ntohl(h.dst_rank);
    h.origin_rank = ntohl(h.origin_rank);

    h.expert_bitmap = network_to_host_u64(h.expert_bitmap);
    h.global_idx = network_to_host_u64(h.global_idx);
    h.timestamp_ns = network_to_host_u64(h.timestamp_ns);

    return h;
}

static bool alloc_aligned_buffer(char*& ptr, size_t size) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        page_size = 4096;
    }

    void* p = nullptr;
    if (posix_memalign(&p, static_cast<size_t>(page_size), size) != 0) {
        return false;
    }

    std::memset(p, 0, size);
    ptr = static_cast<char*>(p);
    return true;
}

static bool post_recv(SrConn* conn) {
    if (!conn || !conn->id || !conn->id->qp || !conn->recv_mr || !conn->recv_buf) {
        return false;
    }

    ibv_sge sge{};
    sge.addr = reinterpret_cast<uintptr_t>(conn->recv_buf);
    sge.length = static_cast<uint32_t>(conn->buf_size);
    sge.lkey = conn->recv_mr->lkey;

    ibv_recv_wr wr{};
    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    ibv_recv_wr* bad_wr = nullptr;
    if (ibv_post_recv(conn->id->qp, &wr, &bad_wr) != 0) {
        std::perror("ibv_post_recv");
        return false;
    }

    conn->recv_posted = true;
    return true;
}

static bool wait_cq(ibv_cq* cq, ibv_wc_opcode expected, uint32_t* byte_len) {
    ibv_wc wc{};

    while (true) {
        int n = ibv_poll_cq(cq, 1, &wc);

        if (n < 0) {
            std::cerr << "ibv_poll_cq failed" << std::endl;
            return false;
        }

        if (n == 0) {
            std::this_thread::yield();
            continue;
        }

        if (wc.status != IBV_WC_SUCCESS) {
            std::cerr << "RDMA work completion failed: "
                      << ibv_wc_status_str(wc.status)
                      << std::endl;
            return false;
        }

        if (wc.opcode != expected) {
            std::cerr << "Unexpected WC opcode: " << wc.opcode << std::endl;
            return false;
        }

        if (byte_len) {
            *byte_len = wc.byte_len;
        }

        return true;
    }
}

static void destroy_conn_resources(SrConn* conn) {
    if (!conn) {
        return;
    }

    if (conn->id) {
        // 如果已经断开，这里失败也没关系。
        ::rdma_disconnect(conn->id);
    }

    if (conn->id && conn->id->qp) {
        rdma_destroy_qp(conn->id);
    }

    if (conn->send_mr) {
        ibv_dereg_mr(conn->send_mr);
        conn->send_mr = nullptr;
    }

    if (conn->recv_mr) {
        ibv_dereg_mr(conn->recv_mr);
        conn->recv_mr = nullptr;
    }

    if (conn->send_cq) {
        ibv_destroy_cq(conn->send_cq);
        conn->send_cq = nullptr;
    }

    if (conn->recv_cq) {
        ibv_destroy_cq(conn->recv_cq);
        conn->recv_cq = nullptr;
    }

    if (conn->pd) {
        ibv_dealloc_pd(conn->pd);
        conn->pd = nullptr;
    }

    if (conn->id) {
        rdma_destroy_id(conn->id);
        conn->id = nullptr;
    }

    if (conn->send_buf) {
        std::free(conn->send_buf);
        conn->send_buf = nullptr;
    }

    if (conn->recv_buf) {
        std::free(conn->recv_buf);
        conn->recv_buf = nullptr;
    }
}

static bool build_conn_resources(SrConn* conn) {
    if (!conn || !conn->id || !conn->id->verbs) {
        std::cerr << "Invalid RDMA connection id" << std::endl;
        return false;
    }

    conn->pd = ibv_alloc_pd(conn->id->verbs);
    if (!conn->pd) {
        std::perror("ibv_alloc_pd");
        return false;
    }

    conn->send_cq = ibv_create_cq(conn->id->verbs, 16, nullptr, nullptr, 0);
    if (!conn->send_cq) {
        std::perror("ibv_create_cq send");
        return false;
    }

    conn->recv_cq = ibv_create_cq(conn->id->verbs, 16, nullptr, nullptr, 0);
    if (!conn->recv_cq) {
        std::perror("ibv_create_cq recv");
        return false;
    }

    if (!alloc_aligned_buffer(conn->send_buf, conn->buf_size)) {
        std::cerr << "Failed to allocate send buffer" << std::endl;
        return false;
    }

    if (!alloc_aligned_buffer(conn->recv_buf, conn->buf_size)) {
        std::cerr << "Failed to allocate recv buffer" << std::endl;
        return false;
    }

    conn->send_mr = ibv_reg_mr(
        conn->pd,
        conn->send_buf,
        conn->buf_size,
        IBV_ACCESS_LOCAL_WRITE
    );

    if (!conn->send_mr) {
        std::perror("ibv_reg_mr send");
        return false;
    }

    conn->recv_mr = ibv_reg_mr(
        conn->pd,
        conn->recv_buf,
        conn->buf_size,
        IBV_ACCESS_LOCAL_WRITE
    );

    if (!conn->recv_mr) {
        std::perror("ibv_reg_mr recv");
        return false;
    }

    ibv_qp_init_attr qp_attr{};
    qp_attr.send_cq = conn->send_cq;
    qp_attr.recv_cq = conn->recv_cq;
    qp_attr.qp_type = IBV_QPT_RC;

    qp_attr.cap.max_send_wr = 16;
    qp_attr.cap.max_recv_wr = 16;
    qp_attr.cap.max_send_sge = 1;
    qp_attr.cap.max_recv_sge = 1;

    if (rdma_create_qp(conn->id, conn->pd, &qp_attr) != 0) {
        std::perror("rdma_create_qp");
        return false;
    }

    // 必须先 post recv，再允许对端 send。
    return post_recv(conn);
}

bool sr_connect(const std::string& addr, uint16_t port, SrConn*& out) {
    out = nullptr;

    rdma_cm_id* id = nullptr;
    if (rdma_create_id(nullptr, &id, nullptr, RDMA_PS_TCP) != 0) {
        std::perror("rdma_create_id");
        return false;
    }

    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* res = nullptr;
    std::string port_str = std::to_string(port);

    int gai = getaddrinfo(addr.c_str(), port_str.c_str(), &hints, &res);
    if (gai != 0 || !res) {
        std::cerr << "getaddrinfo failed: " << gai_strerror(gai) << std::endl;
        rdma_destroy_id(id);
        return false;
    }

    if (rdma_resolve_addr(id, nullptr, res->ai_addr, 2000) != 0) {
        std::perror("rdma_resolve_addr");
        freeaddrinfo(res);
        rdma_destroy_id(id);
        return false;
    }

    freeaddrinfo(res);

    if (rdma_resolve_route(id, 2000) != 0) {
        std::perror("rdma_resolve_route");
        rdma_destroy_id(id);
        return false;
    }

    SrConn* conn = new SrConn();
    conn->id = id;

    if (!build_conn_resources(conn)) {
        destroy_conn_resources(conn);
        delete conn;
        return false;
    }

    rdma_conn_param param{};
    param.initiator_depth = 1;
    param.responder_resources = 1;
    param.retry_count = 7;
    param.rnr_retry_count = 7;

    if (::rdma_connect(id, &param) != 0) {
        std::perror("rdma_connect");
        destroy_conn_resources(conn);
        delete conn;
        return false;
    }

    out = conn;
    return true;
}

bool sr_listen(uint16_t port, SrListener*& out) {
    out = nullptr;

    rdma_cm_id* listen_id = nullptr;
    if (rdma_create_id(nullptr, &listen_id, nullptr, RDMA_PS_TCP) != 0) {
        std::perror("rdma_create_id listen");
        return false;
    }

    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_addr.s_addr = INADDR_ANY;
    local.sin_port = htons(port);

    if (rdma_bind_addr(listen_id, reinterpret_cast<sockaddr*>(&local)) != 0) {
        std::perror("rdma_bind_addr");
        rdma_destroy_id(listen_id);
        return false;
    }

    if (rdma_listen(listen_id, 8) != 0) {
        std::perror("rdma_listen");
        rdma_destroy_id(listen_id);
        return false;
    }

    SrListener* listener = new SrListener();
    listener->id = listen_id;
    out = listener;
    return true;
}

bool sr_accept(SrListener* listener, SrConn*& out) {
    out = nullptr;

    if (!listener || !listener->id) {
        return false;
    }

    rdma_cm_id* id = nullptr;

    // 同步模式下，rdma_get_request 会阻塞直到有连接请求。
    if (rdma_get_request(listener->id, &id) != 0) {
        std::perror("rdma_get_request");
        return false;
    }

    SrConn* conn = new SrConn();
    conn->id = id;

    if (!build_conn_resources(conn)) {
        destroy_conn_resources(conn);
        delete conn;
        return false;
    }

    rdma_conn_param param{};
    param.initiator_depth = 1;
    param.responder_resources = 1;
    param.retry_count = 7;
    param.rnr_retry_count = 7;

    if (::rdma_accept(id, &param) != 0) {
        std::perror("rdma_accept");
        destroy_conn_resources(conn);
        delete conn;
        return false;
    }

    out = conn;
    return true;
}

bool sr_write(SrConn* conn, const MsgHeader& header, const void* payload) {
    if (!conn || !conn->id || !conn->id->qp || !conn->send_buf || !conn->send_mr) {
        return false;
    }

    if (header.payload_len > 0 && payload == nullptr) {
        return false;
    }

    size_t total_len = sizeof(MsgHeader) + static_cast<size_t>(header.payload_len);

    if (total_len > conn->buf_size) {
        std::cerr << "Message too large: " << total_len
                  << " > " << conn->buf_size << std::endl;
        return false;
    }

    if (total_len > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    MsgHeader wire = header_to_wire(header);

    std::memcpy(conn->send_buf, &wire, sizeof(MsgHeader));

    if (header.payload_len > 0) {
        std::memcpy(
            conn->send_buf + sizeof(MsgHeader),
            payload,
            header.payload_len
        );
    }

    ibv_sge sge{};
    sge.addr = reinterpret_cast<uintptr_t>(conn->send_buf);
    sge.length = static_cast<uint32_t>(total_len);
    sge.lkey = conn->send_mr->lkey;

    ibv_send_wr wr{};
    wr.wr_id = 2;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;

    ibv_send_wr* bad_wr = nullptr;

    if (ibv_post_send(conn->id->qp, &wr, &bad_wr) != 0) {
        std::perror("ibv_post_send");
        return false;
    }

    return wait_cq(conn->send_cq, IBV_WC_SEND, nullptr);
}

bool sr_read(SrConn* conn, MsgHeader& header, std::vector<char>& payload) {
    payload.clear();

    if (!conn || !conn->recv_buf || !conn->recv_mr || !conn->recv_cq) {
        return false;
    }

    if (!conn->recv_posted) {
        if (!post_recv(conn)) {
            return false;
        }
    }

    uint32_t byte_len = 0;

    if (!wait_cq(conn->recv_cq, IBV_WC_RECV, &byte_len)) {
        return false;
    }

    conn->recv_posted = false;

    if (byte_len < sizeof(MsgHeader)) {
        std::cerr << "Received message too small: " << byte_len << std::endl;
        return false;
    }

    MsgHeader wire{};
    std::memcpy(&wire, conn->recv_buf, sizeof(MsgHeader));
    header = header_from_wire(wire);

    if (header.magic != MAGIC || header.version != PROTOCOL_VERSION) {
        std::cerr << "Invalid protocol header" << std::endl;
        return false;
    }

    size_t expected_len = sizeof(MsgHeader) + static_cast<size_t>(header.payload_len);

    if (expected_len > byte_len) {
        std::cerr << "Truncated RDMA message: expected "
                  << expected_len << ", got " << byte_len << std::endl;
        return false;
    }

    if (header.payload_len > 0) {
        const char* begin = conn->recv_buf + sizeof(MsgHeader);
        const char* end = begin + header.payload_len;
        payload.assign(begin, end);
    }

    // 为下一条消息提前 post recv。
    return post_recv(conn);
}

void sr_close(SrConn* conn) {
    if (!conn) {
        return;
    }

    destroy_conn_resources(conn);
    delete conn;
}

void sr_close_listener(SrListener* listener) {
    if (!listener) {
        return;
    }

    if (listener->id) {
        rdma_destroy_id(listener->id);
        listener->id = nullptr;
    }

    delete listener;
}

uint64_t ns_timestamp() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}