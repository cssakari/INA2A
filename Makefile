CXX = g++
CXXFLAGS = -std=c++17 -Iinclude -Wall -Wextra -pthread
LDFLAGS = -lrdmacm -libverbs

BIN_DIR = bin

all: $(BIN_DIR)/host_main $(BIN_DIR)/switch_main

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/host_main: src/host_main.cc src/transport_softroce.cc src/workload.cc | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/switch_main: src/switch_main.cc src/transport_softroce.cc | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)