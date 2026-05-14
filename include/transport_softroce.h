#pragma once

#include "protocol.h"

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>

struct SrConn;
struct SrListener;

constexpr size_t SR_MAX_MSG_SIZE = 8 * 1024 * 1024;

bool sr_connect(const std::string& addr, uint16_t port, SrConn*& conn);
bool sr_listen(uint16_t port, SrListener*& listener);
bool sr_accept(SrListener* listener, SrConn*& conn);

bool sr_write(SrConn* conn, const MsgHeader& header, const void* payload);
bool sr_read(SrConn* conn, MsgHeader& header, std::vector<char>& payload);

void sr_close(SrConn* conn);
void sr_close_listener(SrListener* listener);

uint64_t ns_timestamp();

enum class SrReadStatus { OK, TIMEOUT, FAILED };

SrReadStatus sr_read_for(
    SrConn* conn,
    MsgHeader& header,
    std::vector<char>& payload,
    std::chrono::milliseconds timeout
);