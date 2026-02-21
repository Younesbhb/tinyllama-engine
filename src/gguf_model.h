#pragma once


#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <iomanip>
#include <iostream>
#include <cstring>

#include "gguf.h"
#include "tokenizer.h"
#include "ops.h"       // for fp16_to_f32 used in dump_tensor

struct MMapFile {
    //file descriptor (Linux/macOS "file handle"). -1 means "not opened".
    int fd = -1;
    //base: pointer to the start of the mapped file in memory.
    std::uint8_t* base = nullptr;
    //size: file size in bytes
    std::uint64_t size = 0;

    explicit MMapFile(const std::string& path);
    ~MMapFile();
    
    //"This object cannot be copied."
    //= delete means "this operation is forbidden"
    //"The copy constructor does not exist."
    MMapFile(const MMapFile&) = delete;
    MMapFile& operator=(const MMapFile&) = delete;
};

struct Cursor {
    const std::uint8_t* p = nullptr;
    const std::uint8_t* end = nullptr;
    std::uint64_t off = 0; // absolute offset from file start. How many bytes we have consumed from the beginning of the file.

    Cursor(const std::uint8_t* base, std::uint64_t size);
    void need(std::uint64_t n) const;
    const std::uint8_t* take(std::uint64_t n);

    template <typename T>
    T read_pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        const std::uint8_t* b = take(sizeof(T));
        T out{};
        std::memcpy(&out, b, sizeof(T));
        return out;
    }

    std::string read_string();
    void skip_string();
};

class GGUFModel {
public:
    explicit GGUFModel(const std::string& path);

    std::uint32_t version() const;
    std::uint64_t n_tensors() const;
    std::uint64_t n_kv() const;
    std::uint64_t alignment() const;
    std::uint64_t tensor_data_start() const;
    std::uint64_t file_size() const;

    const gguf_tensor_info_t& tensor_info(const std::string& name) const;
    const std::uint8_t* tensor_bytes(const gguf_tensor_info_t& t) const;
    const llama_config_t& config() const;
    const Tokenizer& tokenizer() const;

    void dump_tensor(const std::string& name, std::size_t n = 10) const;

private:

    MMapFile mm_;
    Cursor cur_;

    std::uint32_t version_ = 0;
    std::uint64_t n_tensors_ = 0;
    std::uint64_t n_kv_ = 0;
    std::uint64_t alignment_ = 32;
    std::uint64_t tensor_data_start_ = 0;

    std::vector<gguf_tensor_info_t> tensors_;
    std::unordered_map<std::string, std::size_t> index_;
    llama_config_t config_;
    Tokenizer tokenizer_;

    void parse();
};