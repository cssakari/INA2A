// log.h （增加 logf 方法）
#ifndef LOG_H
#define LOG_H

#include <iostream>
#include <fstream>
#include <streambuf>
#include <memory>
#include <ctime>
#include <mutex>
#include <string>
#include <sstream>
#include <iomanip>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <cstdarg>

class Logger {
public:
    enum Level { INFO, WARN, ERROR };

    Logger(const std::string& filename, bool alsoToConsole = false)
        : toConsole(alsoToConsole), stopFlag(false)
    {
        logFile.open(filename, std::ios::app);
        if (!logFile.is_open()) {
            throw std::runtime_error("无法打开日志文件: " + filename);
        }

        oldCoutBuf = std::cout.rdbuf();
        oldCerrBuf = std::cerr.rdbuf();

        coutStreamBuf = std::make_unique<LogStreamBuf>(*this, oldCoutBuf, toConsole, INFO);
        cerrStreamBuf = std::make_unique<LogStreamBuf>(*this, oldCerrBuf, toConsole, ERROR);

        std::cout.rdbuf(coutStreamBuf.get());
        std::cerr.rdbuf(cerrStreamBuf.get());

        worker = std::thread(&Logger::processQueue, this);
    }

    ~Logger() {
        std::cout.flush();
        std::cerr.flush();

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            stopFlag = true;
        }
        cv.notify_all();
        if (worker.joinable()) worker.join();

        std::cout.rdbuf(oldCoutBuf);
        std::cerr.rdbuf(oldCerrBuf);
        logFile.close();
    }

    // 流式 log
    void log(Level level, const std::string& message) {
        std::ostringstream oss;
        oss << "[" << currentDateTime() << "]"
            << "[" << levelToString(level) << "] "
            << message << "\n";

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            logQueue.push({level, oss.str()});
        }
        cv.notify_one();
    }

    // 新增：printf 风格 logf
    void logf(Level level, const char* fmt, ...) {
        char buf[1024];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        log(level, buf);
    }

private:
    struct LogItem {
        Level level;
        std::string text;
    };

    std::ofstream logFile;
    bool toConsole;
    std::streambuf* oldCoutBuf;
    std::streambuf* oldCerrBuf;
    std::unique_ptr<std::streambuf> coutStreamBuf;
    std::unique_ptr<std::streambuf> cerrStreamBuf;

    std::queue<LogItem> logQueue;
    std::mutex queueMutex;
    std::condition_variable cv;
    std::thread worker;
    std::atomic<bool> stopFlag;

    class LogStreamBuf : public std::streambuf {
    public:
        LogStreamBuf(Logger& logger, std::streambuf* consoleBuf, bool toConsole, Level defaultLevel)
            : logger(logger), consoleBuf(consoleBuf), toConsole(toConsole), level(defaultLevel) {}

    protected:
        virtual int overflow(int c) override {
            if (c != EOF) {
                char ch = static_cast<char>(c);
                buffer += ch;
                if (ch == '\n') {
                    logger.log(level, buffer.substr(0, buffer.size()-1));
                    buffer.clear();
                }
            }
            return c;
        }

        virtual int sync() override {
            if (!buffer.empty()) {
                logger.log(level, buffer);
                buffer.clear();
            }
            return 0;
        }

    private:
        Logger& logger;
        std::streambuf* consoleBuf;
        bool toConsole;
        Level level;
        std::string buffer;
    };

    void processQueue() {
        while (true) {
            LogItem item;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                cv.wait(lock, [this]() { return !logQueue.empty() || stopFlag; });
                if (logQueue.empty() && stopFlag) break;
                item = logQueue.front();
                logQueue.pop();
            }

            logFile << item.text;
            logFile.flush();

            if (toConsole) {
                if (item.level == ERROR)
                    oldCerrBuf->sputn(item.text.c_str(), item.text.size());
                else
                    oldCoutBuf->sputn(item.text.c_str(), item.text.size());
            }
        }
    }

    std::string currentDateTime() {
        auto t = std::time(nullptr);
        std::tm tm{};
#if defined(_WIN32) || defined(_WIN64)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    std::string levelToString(Level level) {
        switch (level) {
            case INFO: return "INFO"; 
            case WARN: return "WARN"; 
            case ERROR: return "ERROR"; 
            default: return "INFO";
        }
    }
};

#endif // LOG_H