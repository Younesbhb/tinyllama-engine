#include <cstdint>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>
#include <algorithm>

#include "gguf.h"
#include "tokenizer.h"
#include "ops.h"
#include "run_state.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iomanip> // for precision output

// Some definitions
// n_embd (2048) — The "width" of the model. Every token gets represented as 2,048 numbers. 
// This vector flows through the entire model.

// n_layers (22) — How many times the token gets processed. Each layer refines the representation.

// n_ff (5632): "feed-forward dimension." The FFN temporarily expands from 2048 to 5632 (bigger workspace to "think" in), then shrinks back.

// n_ctx (2048): "context length." Maximum number of tokens the model can see at once — prompt plus generated text combined.

// n_vocab (32000): "vocabulary size." How many words/tokens the model knows.

// n_layers (22): How many transformer blocks the signal passes through.

// kv_dim (256) — The size of the key and value vectors. Smaller than 2048 because of 
// GQA (4 KV heads × 64 dimensions per head = 256 instead of 32 × 64 = 2048).

// Input embedding matrix (token_embd.weight): Size 32,000 × 2,048. Used at the very start of the forward pass. Converts a token ID into a 2,048-number vector. Row 15043 is the vector for "Hello." We used this in embed_token().

// Output matrix (output.weight): Size 32,000 × 2,048. Used at the very end. Converts the final 2,048-number hidden state into 32,000 scores — one per vocabulary word.

// n_head (32): The number of query heads. The model runs attention 32 times in parallel, each focusing on different aspects of the input. One head might focus on grammar, another on meaning, another on nearby words, etc.

// n_head_kv (4): The number of key/value heads. Instead of each query head having its own key and value (which would need 32 KV heads), multiple query heads share KV heads. 32 query heads share 4 KV heads — that's 8 query heads per KV head. This saves memory.

// kv_dim (256): The total size of the key and value vectors. It's n_head_kv × head_dim = 4 × 64 = 256. This isn't stored in the config — we compute it in the forward pass because it's derived from other values.

// head_dim (64): The size of each head's vector. It's calculated as n_embd / n_head = 2048 / 32 = 64. Each head works on a 64-dimensional slice. All 32 heads together: 32 × 64 = 2048 (the full hidden state).

// What is normalizing?
// Imagine you have the numbers [1000, 2000, 3000]. After multiplying through several layers, they might become [5000000, 10000000, 15000000]. After a few more layers, they could overflow to infinity.
// Normalizing scales them back to a reasonable range. RMSNorm takes those giant numbers and rescales them so the average magnitude is around 1.0:
//
//Do we have a KV cache for each layer? A query cache?
// KV cache: yes, one per layer. The cache stores keys and values for all 22 layers, each with space for 2048 positions × 256 floats.
// Query cache: no. Queries are never cached. A query is only used once — at the moment the token is processed — to compute attention scores against all cached keys. After that, the query is thrown away.

// -------------------- small helpers --------------------

static std::string sys_err(const char* what) {
    return std::string(what) + ": " + std::strerror(errno);
}

static std::uint64_t align_up(std::uint64_t x, std::uint64_t a) {
    if (a == 0) throw std::runtime_error("alignment is 0");
    std::uint64_t r = x % a;
    return (r == 0) ? x : (x + (a - r));
}

static const char* ggml_type_name(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        default: return "OTHER";
    }
}

static std::uint64_t bytes_per_element(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_F16: return 2;
        default:
            throw std::runtime_error("Unsupported ggml_type for now (only F16/F32 supported).");
    }
}

// -------------------- mmap RAII --------------------
// "RAII" means cleanup happens automatically.

struct MMapFile {
    //file descriptor (Linux/macOS "file handle"). -1 means "not opened".
    int fd = -1;
    //base: pointer to the start of the mapped file in memory.
    std::uint8_t* base = nullptr;
    //size: file size in bytes
    std::uint64_t size = 0;

    explicit MMapFile(const std::string& path) {
        //1) Open the file (read-only)
        
        //path.c_str() converts a C++ std::string into a C-style string (const char*)
        //O_RDONLY = open for reading only.
        //if it fails -> throw error.
        //fd is just an ID number that the operating system uses to look up the real file information inside the kernel.
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error(sys_err("open"));

        //2) Get file size

        /*
        fstat fills st with info about the file (including size).
        If error or size 0 -> close file and throw.
        */
        // "Using this file descriptor fd, fill st with info about the file."
        //stat is a struct the OS uses to return file info (size, permissions, etc.).
        //"Hey OS, look up the file associated with descriptor fd and copy its info into st."
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error(sys_err("fstat"));
        }
        if (st.st_size <= 0) {
            ::close(fd);
            throw std::runtime_error("file size is zero");
        }
        size = static_cast<std::uint64_t>(st.st_size);

        
        //Map this file into memory for reading.
        /*
        nullptr = OS chooses where to map it.
        size = map whole file
        PROT_READ = read-only memory
        MAP_PRIVATE = changes (if any) won't write back to disk
        fd = which file
        0 = AKA Offset into the file (0 means start at beginning).
        */
        void* p = ::mmap(nullptr, static_cast<size_t>(size), PROT_READ, MAP_PRIVATE, fd, 0);
        
        //If mapping fails -> close fd and throw.
        if (p == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error(sys_err("mmap"));
        }
        base = static_cast<std::uint8_t*>(p);
        /*
        base points to byte 0 of the file
        base[i] gives you the i-th byte
        No read() calls needed; OS loads pages as you access them.
        */
    }


    //"This object cannot be copied."
    //= delete means "this operation is forbidden"
    //"The copy constructor does not exist."
    MMapFile(const MMapFile&) = delete;
    MMapFile& operator=(const MMapFile&) = delete;

    ~MMapFile() {
        if (base && base != MAP_FAILED) {
            //unmap the memory (munmap)
            ::munmap(base, static_cast<size_t>(size));
        }
        if (fd >= 0) ::close(fd);
    }
};


// -------------------- safe cursor over mapped bytes --------------------

struct Cursor {
    const std::uint8_t* p = nullptr;
    const std::uint8_t* end = nullptr;
    std::uint64_t off = 0; // absolute offset from file start. How many bytes we have consumed from the beginning of the file.

    Cursor(const std::uint8_t* base, std::uint64_t size) : p(base), end(base + size), off(0) {}

    void need(std::uint64_t n) const {
        if (static_cast<std::uint64_t>(end - p) < n) {
            throw std::runtime_error("Unexpected EOF while parsing");
        }
    }

    const std::uint8_t* take(std::uint64_t n) {
        //n is the nb of bytes
        need(n);
        const std::uint8_t* out = p;
        p += n;
        off += n;
        return out;
    }

    template <typename T>
    T read_pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        const std::uint8_t* b = take(sizeof(T));
        T out{};
        std::memcpy(&out, b, sizeof(T));
        return out;
    }

    std::string read_string() {
        std::uint64_t len = read_pod<std::uint64_t>();
        if (len > (1ull << 30)) throw std::runtime_error("Suspicious string length (>1GB)");
        const std::uint8_t* b = take(len);
        return std::string(reinterpret_cast<const char*>(b), static_cast<size_t>(len));
    }

    void skip_string() {
        std::uint64_t len = read_pod<std::uint64_t>();
        //we can do that since len represents the length (nb of char == nb of bytes) in the string
        take(len);
    }
};

// -------------------- metadata skipping (so we reach tensor infos safely) --------------------

static std::uint64_t gguf_elem_size(gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8:   return 1;
        case GGUF_TYPE_INT8:    return 1;
        case GGUF_TYPE_UINT16:  return 2;
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:  return 4;
        case GGUF_TYPE_INT32:   return 4;
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_BOOL:    return 1; 
        case GGUF_TYPE_UINT64:  return 8;
        case GGUF_TYPE_INT64:   return 8;
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static void skip_value(Cursor& c, gguf_type t, int depth = 0) {
    if (depth > 8) throw std::runtime_error("Metadata array nesting too deep");

    switch (t) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:    c.take(1); return;
        
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:   c.take(2); return;
        
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: c.take(4); return;
        
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: c.take(8); return;
        
        case GGUF_TYPE_STRING:  c.skip_string(); return;

        case GGUF_TYPE_ARRAY: {
            //Array elem type
            gguf_type elem_t = static_cast<gguf_type>(c.read_pod<std::uint32_t>());
            //Array length
            std::uint64_t n  = c.read_pod<std::uint64_t>();

            if (elem_t >= GGUF_TYPE_COUNT) throw std::runtime_error("Array elem_type out of range");

            if (elem_t == GGUF_TYPE_STRING) {
                for (std::uint64_t i = 0; i < n; i++) c.skip_string();
                return;
            }

            if (elem_t == GGUF_TYPE_ARRAY) {
                // Rare but possible: array-of-arrays
                for (std::uint64_t i = 0; i < n; i++) {
                    skip_value(c, GGUF_TYPE_ARRAY, depth + 1);
                }
                return;
            }
            
            //Byte size per element
            std::uint64_t esz = gguf_elem_size(elem_t);
            if (esz == 0) throw std::runtime_error("Invalid elem size for array");
            if (n != 0 && esz > std::numeric_limits<std::uint64_t>::max() / n)
                throw std::runtime_error("Array byte size overflow");
            //Skip the array 
            c.take(n * esz);
            return;
        }

        default:
            throw std::runtime_error("Unknown GGUF metadata type");
    }
}

// -------------------- FP16 -> float --------------------

static float fp16_to_f32(std::uint16_t h) {
    std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    std::uint32_t exp  = (static_cast<std::uint32_t>(h) >> 10) & 0x1Fu;
    std::uint32_t mant =  static_cast<std::uint32_t>(h) & 0x3FFu;

    std::uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3FFu;
            std::uint32_t fexp  = static_cast<std::uint32_t>(e + 127);
            std::uint32_t fmant = mant << 13;
            bits = sign | (fexp << 23) | fmant;
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
        // Make NaN quiet (optional but good practice)
        if (mant != 0){
            bits |= 0x00400000u;
        }
    } else {
        std::uint32_t fexp  = exp + (127 - 15);
        std::uint32_t fmant = mant << 13;
        bits = sign | (fexp << 23) | fmant;
    }

    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}




// -------------------- Model class (owns mmap + parsed index) --------------------

class GGUFModel {
public:
    explicit GGUFModel(const std::string& path)
        : mm_(path), cur_(mm_.base, mm_.size) {
        parse();
    }

    // const here means calling this function will NOT modify the object.
    std::uint32_t version() const { return version_; }
    std::uint64_t n_tensors() const { return n_tensors_; }
    std::uint64_t n_kv() const { return n_kv_; }
    std::uint64_t alignment() const { return alignment_; }
    std::uint64_t tensor_data_start() const { return tensor_data_start_; }
    std::uint64_t file_size() const { return mm_.size; }

    const gguf_tensor_info_t& tensor_info(const std::string& name) const {
        auto it = index_.find(name);
        if (it == index_.end()) throw std::runtime_error("Tensor not found: " + name);
        return tensors_.at(it->second);
    }

    const std::uint8_t* tensor_bytes(const gguf_tensor_info_t& t) const {
        // abs_offset computed during parse + sanity checks
        return mm_.base + t.abs_offset;
    }

    const llama_config_t& config() const{
        return config_;
    }
    
    const Tokenizer& tokenizer() const {return tokenizer_;}




    void dump_tensor(const std::string& name, std::size_t n = 10) const {
        const auto& t = tensor_info(name);
        const std::uint8_t* data = tensor_bytes(t);
        std::cout << std::setprecision(10);

        std::cout << "Tensor: " << t.name << "\n"
                  << "  type: " << ggml_type_name(t.type) << "\n"
                  << "  dims: [";
        for (std::size_t i = 0; i < t.dimensions.size(); i++) {
            std::cout << t.dimensions[i] << (i + 1 == t.dimensions.size() ? "" : ", ");
        }
        std::cout << "]\n"
                  << "  offset(rel): " << t.offset << "\n"
                  << "  abs_offset:  " << t.abs_offset << "\n"
                  << "  byte_size:   " << t.byte_size << "\n"
                  << "  first " << n << " values: ";

        n = std::min<std::size_t>(n, static_cast<std::size_t>(t.n_elems));

        if (t.type == GGML_TYPE_F32) {
            for (std::size_t i = 0; i < n; i++) {
                float v;
                std::memcpy(&v, data + i * 4, 4);
                std::cout << v << " ";
            }
        } else if (t.type == GGML_TYPE_F16) {
            for (std::size_t i = 0; i < n; i++) {
                std::uint16_t h;
                std::memcpy(&h, data + i * 2, 2);
                std::cout << fp16_to_f32(h) << " ";
            }
        } else {
            std::cout << "(unsupported type for dump right now)";
        }
        std::cout << "\n";
    }

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



    void parse() {
        // magic
        const std::uint8_t* m = cur_.take(4);
        if (!(m[0]=='G' && m[1]=='G' && m[2]=='U' && m[3]=='F')) {
            throw std::runtime_error("Not a GGUF file");
        }

        version_   = cur_.read_pod<std::uint32_t>();
        n_tensors_ = cur_.read_pod<std::uint64_t>();
        n_kv_      = cur_.read_pod<std::uint64_t>();

        std::vector<std::string> vocab_temp;
        std::vector<float> scores_temp;


        //KV metadata loop 
        for (std::uint64_t i = 0; i < n_kv_ ; i++){
            //Key name
            std::string key = cur_.read_string();
            //Value type
            std::uint32_t val_type = cur_.read_pod<std::uint32_t>();
            if (val_type >= GGUF_TYPE_COUNT) throw std::runtime_error("KV type out of range");
            gguf_type type = static_cast<gguf_type>(val_type);
            
            
            if(key == "general.alignment"){
                if (type == GGUF_TYPE_UINT32) alignment_ = cur_.read_pod<uint32_t>();
                else if (type == GGUF_TYPE_UINT64) {
                    alignment_ = static_cast<std::uint32_t>(cur_.read_pod<uint64_t>());
                }
                else skip_value(cur_, type);
            }
            else if (key == "general.architecture" && type == GGUF_TYPE_STRING){
                config_.architecture = cur_.read_string();
            }
            else if (key == "llama.block_count" || key == "llama.layer_count"){
                if(type == GGUF_TYPE_UINT32) config_.n_layers = cur_.read_pod<uint32_t>();
                else if (type == GGUF_TYPE_UINT64){ 
                    config_.n_layers = static_cast<std::uint32_t>(cur_.read_pod<std::uint64_t>());
                }
                else skip_value(cur_,type);
            }
            else if (key == "llama.embedding_length"){
                if (type == GGUF_TYPE_UINT32) config_.n_embd = cur_.read_pod<uint32_t>();
                else if (type == GGUF_TYPE_UINT64){ 
                    config_.n_embd = static_cast<std::uint32_t>(cur_.read_pod<std::uint64_t>());
                }
                else skip_value(cur_,type);
            }
            else if (key == "llama.attention.head_count") {
                if (type == GGUF_TYPE_UINT32) config_.n_head = cur_.read_pod<std::uint32_t>();
                else if (type == GGUF_TYPE_UINT64){ 
                    config_.n_head = static_cast<std::uint32_t>(cur_.read_pod<std::uint64_t>());
                }
                else skip_value(cur_,type);
            }
            else if (key == "llama.attention.head_count_kv") {
                if (type == GGUF_TYPE_UINT32) config_.n_head_kv = cur_.read_pod<std::uint32_t>();
                else if (type == GGUF_TYPE_UINT64){ 
                    config_.n_head_kv = static_cast<std::uint32_t>(cur_.read_pod<std::uint64_t>());
                }
                else skip_value(cur_,type);
            }
            else if (key == "llama.feed_forward_length") {
                if (type == GGUF_TYPE_UINT32) config_.n_ff = cur_.read_pod<std::uint32_t>();
                else if (type == GGUF_TYPE_UINT64){ 
                    config_.n_ff = static_cast<std::uint32_t>(cur_.read_pod<std::uint64_t>());
                    }
                else skip_value(cur_,type);
            }
            else if (key == "llama.rope.dimension_count") {
                if (type == GGUF_TYPE_UINT32) config_.rope_dim = cur_.read_pod<std::uint32_t>();
                else if (type == GGUF_TYPE_UINT64){ 
                    config_.rope_dim = static_cast<std::uint32_t>(cur_.read_pod<std::uint64_t>());
                }
                else skip_value(cur_,type);
            }
            else if (key == "llama.rope.freq_base") {
                if (type == GGUF_TYPE_FLOAT32) config_.rope_freq_base = cur_.read_pod<float>();
                else skip_value(cur_,type);
            }
            else if (key == "llama.context_length") {
                if (type == GGUF_TYPE_UINT32) config_.n_ctx = cur_.read_pod<std::uint32_t>();
                else if (type == GGUF_TYPE_UINT64){
                    config_.n_ctx = static_cast<std::uint32_t>(cur_.read_pod<std::uint64_t>());
                }
                else skip_value(cur_,type);
            }
            else if (key == "llama.attention.layer_norm_rms_epsilon") {
                if (type == GGUF_TYPE_FLOAT32) config_.rms_norm_eps = cur_.read_pod<float>();
                else skip_value(cur_,type);
            }
            else if(key == "tokenizer.ggml.tokens"){
                if(type == GGUF_TYPE_ARRAY){
                    //Read array type
                    std::uint32_t elem_type = cur_.read_pod<std::uint32_t>();
                    if (elem_type >= GGUF_TYPE_COUNT) throw std::runtime_error("Array type of tokenizer.ggml.tokens out of range.");
                    gguf_type array_type = static_cast<gguf_type>(elem_type);
                    if(array_type != GGUF_TYPE_STRING) throw std::runtime_error("Array type is not a string.");
                    
                    std::uint64_t length = cur_.read_pod<std::uint64_t>();

                    vocab_temp.reserve(static_cast<std::size_t>(length));
                    for (uint64_t j = 0; j  < length; j++){
                        vocab_temp.push_back(cur_.read_string());
                    }
                    
                } else throw std::runtime_error("tokenizer.ggml.tokens is expected to be an array but couldnt be read as one");
            }
            else if (key == "tokenizer.ggml.scores"){
                if(type == GGUF_TYPE_ARRAY){
                    std::uint32_t elem_type = cur_.read_pod<std::uint32_t>();
                    if (elem_type >= GGUF_TYPE_COUNT) throw std::runtime_error("Array type of tokenizer.ggml.scores out of range.");
                    gguf_type array_type = static_cast<gguf_type>(elem_type);
                    if(array_type != GGUF_TYPE_FLOAT32) throw std::runtime_error("Array type of tokenizer.ggml.scores is not a 32bit float");

                    std::uint64_t length = cur_.read_pod<std::uint64_t>();
                    scores_temp.reserve(static_cast<std::size_t>(length));
                    for (uint64_t j = 0; j < length; j++){
                        scores_temp.push_back(cur_.read_pod<float>());
                    }
                }
                else throw  std::runtime_error("tokenizer.ggml.scores is expected to be an array but couldnt be read as one");
            }
            else{
                skip_value(cur_,type);
            }
        }
        

        if (alignment_ == 0) throw std::runtime_error("general.alignment is 0");
        if (alignment_ > 4096) throw std::runtime_error("Suspicious alignment (>4096)");
        if (config_.n_layers == 0) throw std::runtime_error("Missing llama.block_count/llama.layer_count in metadata");
        if (config_.n_embd == 0) throw std::runtime_error("Missing llama.embedding_length in metadata");
        if (config_.n_head == 0) throw std::runtime_error("Missing llama.attention.head_count in metadata");

        if(!vocab_temp.empty() && !scores_temp.empty()){
            tokenizer_.load(std::move(vocab_temp),std::move(scores_temp));
        }


        // tensor infos
        tensors_.reserve(static_cast<std::size_t>(n_tensors_));
        index_.reserve(static_cast<std::size_t>(n_tensors_));

        // Store tensor info into the vector and unordered map
        for (std::uint64_t i = 0; i < n_tensors_; i++) {
            gguf_tensor_info_t t;
            t.name = cur_.read_string();
            t.n_dimensions = cur_.read_pod<std::uint32_t>();

            if (t.n_dimensions == 0) throw std::runtime_error("Tensor has 0 dims: " + t.name);
            if (t.n_dimensions > 10) throw std::runtime_error("Suspicious n_dimensions: " + t.name);

            t.dimensions.reserve(static_cast<std::size_t>(t.n_dimensions));
            for (std::uint32_t d = 0; d < t.n_dimensions; d++) {
                t.dimensions.push_back(cur_.read_pod<std::uint64_t>());
            }

            std::uint32_t ty = cur_.read_pod<std::uint32_t>();
            if (ty >= GGML_TYPE_COUNT) throw std::runtime_error("Tensor type out of range: " + t.name);
            t.type = static_cast<ggml_type>(ty);

            t.offset = cur_.read_pod<std::uint64_t>();

            std::size_t idx = tensors_.size();
            auto [it, inserted] = index_.emplace(t.name, idx);
            if (!inserted) throw std::runtime_error("Duplicate tensor name: " + t.name);

            tensors_.push_back(std::move(t));
        }


        // Infer vocab size from tensor "token_embd.weight"
        gguf_tensor_info_t token_embd_weight_t = tensor_info("token_embd.weight");
        if(token_embd_weight_t.dimensions.size() >= 2){
            config_.n_vocab = static_cast<std::uint32_t>(token_embd_weight_t.dimensions[1]);
        }
         else {
            throw std::runtime_error("token_embd.weight has unexpected dimensions");
        }
        // Default n_head_kv to n_head if not specified (non-GQA models)
        if (config_.n_head_kv == 0) {
            config_.n_head_kv = config_.n_head;
        }


        // tensor_data_start = end_of_tensor_infos aligned up
        tensor_data_start_ = align_up(cur_.off, alignment_);
        if (tensor_data_start_ > mm_.size) throw std::runtime_error("tensor_data_start beyond file size");



        // sanity-check tensors + compute abs offsets + sizes
        for (auto& t : tensors_) {
            // compute n_elems i.e. the number of weights
            std::uint64_t n = 1;
            for (std::uint64_t d : t.dimensions) {
                if (d == 0) throw std::runtime_error("Zero dimension tensor: " + t.name);
                if (n > std::numeric_limits<std::uint64_t>::max() / d)
                    throw std::runtime_error("n_elems overflow: " + t.name);
                n *= d;
            }
            t.n_elems = n;

            // compute byte_size
            std::uint64_t bpe = bytes_per_element(t.type);
            if (n > std::numeric_limits<std::uint64_t>::max() / bpe)
                throw std::runtime_error("byte_size overflow: " + t.name);
            t.byte_size = n * bpe;

            // alignment check (offset is relative to tensor_data_start)
            if (t.offset % alignment_ != 0) {
                throw std::runtime_error("Unaligned tensor offset: " + t.name);
            }

            // abs_offset
            if (tensor_data_start_ > std::numeric_limits<std::uint64_t>::max() - t.offset)
                throw std::runtime_error("abs_offset overflow: " + t.name);
            t.abs_offset = tensor_data_start_ + t.offset;

            // bounds: [abs_offset, abs_offset + byte_size] within file
            if (t.abs_offset > mm_.size) throw std::runtime_error("Tensor start beyond file: " + t.name);
            if (t.byte_size > std::numeric_limits<std::uint64_t>::max() - t.abs_offset)
                throw std::runtime_error("Tensor end overflow: " + t.name);

            std::uint64_t end = t.abs_offset + t.byte_size;
            if (end > mm_.size) throw std::runtime_error("Tensor out of file bounds: " + t.name);
        }
        


    }
};

// -------------------- Forward Pass Helpers --------------------


// embed_token(out, model, token) — Goes to the embedding table (a big grid of 32,000 rows × 2,048 columns stored in the model file) and 
// copies row number 'token' into 'out'. If token is 15043, it copies the 15,043rd row — 2,048 numbers that represent what "Hello" means to the model.
// These numbers were learned during training.
static void embed_token(float* out, const GGUFModel& model, int token) {
    const auto& t = model.tensor_info("token_embd.weight");
    const std::uint8_t* data = model.tensor_bytes(t);
    int n_embd = static_cast<int>(model.config().n_embd);

    if (t.type == GGML_TYPE_F16) {
        const std::uint16_t* emb = reinterpret_cast<const std::uint16_t*>(data);
        const std::uint16_t* row = emb + static_cast<std::size_t>(token) * n_embd;
        for (int i = 0; i < n_embd; i++) {
            out[i] = fp16_to_f32(row[i]);
        }
    } else if (t.type == GGML_TYPE_F32) {
        const float* emb = reinterpret_cast<const float*>(data);
        const float* row = emb + static_cast<std::size_t>(token) * n_embd;
        for (int i = 0; i < n_embd; i++) {
            out[i] = row[i];
        }
    } else {
        throw std::runtime_error("Unsupported embedding type");
    }
}

// Looks up a small array of 2,048 numbers from the model file by name. 
// These are the learned weights for RMSNorm 
static const float* get_norm_weight(const GGUFModel& model, const std::string& name) {
    const auto& t = model.tensor_info(name);
    if (t.type != GGML_TYPE_F32) {
        throw std::runtime_error("Expected F32 norm weight: " + name);
    }
    return reinterpret_cast<const float*>(model.tensor_bytes(t));
}

struct WeightRef {
    const void* data;
    ggml_type type;
};

// Looks up a weight matrix from the model file by name. 
// Returns both the pointer to the data and what format it's in (F16 or F32), 
// so matmul knows how to read it.
static WeightRef get_weight(const GGUFModel& model, const std::string& name) {
    const auto& t = model.tensor_info(name);
    return { model.tensor_bytes(t), t.type };
}

// -------------------- Forward Pass --------------------
// model — The model file. Contains all the weight matrices (the "knowledge" the model learned during training). Read-only, never changes.
// state — Scratch space. Contains temporary buffers (x, xb, q, k, v, etc.) and the KV cache. Gets overwritten every call.
// token — The token ID to process (like 15043 for "Hello").
// pos — Where this token sits in the sequence (0, 1, 2, ...). Used by RoPE to encode word order.

static void forward(GGUFModel& model, RunState& state, int token, int pos) {
    const auto& cfg = model.config();
    int n_embd   = static_cast<int>(cfg.n_embd);
    int n_layers = static_cast<int>(cfg.n_layers);
    int n_ff     = static_cast<int>(cfg.n_ff);
    int kv_dim   = static_cast<int>(cfg.n_head_kv * cfg.head_dim());

    // Convert the token ID into a vector of 2,048 numbers
    embed_token(state.x, model, token);

    // Run for 22 layers
    for (int l = 0; l < n_layers; l++) {
        std::string prefix = "blk." + std::to_string(l) + ".";

        const float* attn_norm_w = get_norm_weight(model, prefix + "attn_norm.weight");
        rmsnorm(state.xb, state.x, attn_norm_w, n_embd, cfg.rms_norm_eps);

        auto wq = get_weight(model, prefix + "attn_q.weight");
        auto wk = get_weight(model, prefix + "attn_k.weight");
        auto wv = get_weight(model, prefix + "attn_v.weight");

        matmul(state.q, wq.data, state.xb, n_embd, n_embd, wq.type);    // fills state.q
        matmul(state.k, wk.data, state.xb, kv_dim,  n_embd, wk.type);   // fills state.k
        matmul(state.v, wv.data, state.xb, kv_dim,  n_embd, wv.type);   // fills state.v

        rope(state.q, state.k, pos,
             static_cast<int>(cfg.head_dim()),
             static_cast<int>(cfg.n_head),
             static_cast<int>(cfg.n_head_kv),
             cfg.rope_freq_base);

        // Stores this token's key and value into the KV cache at position pos in layer l
        // Compares this token's query against every previous token's cached key (dot product)
        // to compute relevance scores.
        // Converts scores to percentages with softmax
        // Blends the cached values using those percentages
        // The result goes into state.xb2 — 2,048 numbers representing "what I learned by looking at all previous tokens."
        // state.att is scratch space the attention function uses internally for the scores.

        attention(state.xb2, state.q, state.k, state.v,
                  state.key_cache, state.value_cache, state.att,
                  l, pos, cfg);

        auto wo = get_weight(model, prefix + "attn_output.weight");
        matmul(state.xb, wo.data, state.xb2, n_embd, n_embd, wo.type);

        vec_add(state.x, state.xb, n_embd);

        // --------------------- FFN (thinking) ---------------------

        const float* ffn_norm_w = get_norm_weight(model, prefix + "ffn_norm.weight");
        rmsnorm(state.xb, state.x, ffn_norm_w, n_embd, cfg.rms_norm_eps);

        auto wgate = get_weight(model, prefix + "ffn_gate.weight");
        auto wup   = get_weight(model, prefix + "ffn_up.weight");
        auto wdown = get_weight(model, prefix + "ffn_down.weight");

        ffn_swiglu(state.xb2, state.xb,
                   state.hb, state.hb2,
                   wgate.data, wup.data, wdown.data,
                   n_ff, n_embd, wgate.type);

        vec_add(state.x, state.xb2, n_embd);
    }
    // After the loop ends, state.x holds the model's full understanding — 2048 numbers encoding everything the model knows about what should come next. But these numbers are in the model's "internal language" — they're meaningful to the model's math but not to us.   
    // After the forward function ends, state.logits holds 32000 human-interpretable scores — one per word in the vocabulary. "the" gets 2.1, "on" gets 8.7, "quietly" gets 5.3, etc.
    // The final norm + matmul is just a translation step. It converts the 2048-number internal representation into 32000 concrete word predictions. 
    // state.x doesn't change in a meaningful way after the loop — it just gets normalized (cleaned up) and then multiplied by the output matrix to produce state.logits. The "thinking" is done. The last step is just reading out the answer.

    // ---- Final Output: Convert hidden state to word predictions ----
    // All 22 layers are done. state.x now holds the final refined 2048-number
    // representation of the current token, enriched with context from all 
    // previous tokens. 
    // 
    // We normalize one last time, then multiply by the output matrix 
    // (32000 × 2048) to produce 32000 scores (logits) — one per word 
    // in the vocabulary. The highest score = the model's best guess 
    // for the next word.
    //
    // TinyLlama uses "tied embeddings": no separate output.weight exists,
    // so we reuse token_embd.weight (the same matrix used to convert 
    // token IDs to vectors at the start).
    const float* final_norm_w = get_norm_weight(model, "output_norm.weight");
    rmsnorm(state.x, state.x, final_norm_w, n_embd, cfg.rms_norm_eps);

    WeightRef w_out;
    bool found_output = false;
    try {
        w_out = get_weight(model, "output.weight");
        found_output = true;
    } catch (...) {}
    if (!found_output) {
        w_out = get_weight(model, "token_embd.weight");
    }

    // By the end, state.x contains the original embedding plus 22 attention contributions plus 22 FFN contributions. 
    // Every layer's work is preserved in that sum.
    matmul(state.logits, w_out.data, state.x,
           static_cast<int>(cfg.n_vocab), n_embd, w_out.type);
}


// -------------------- Generation Loop --------------------
//
// This is where the engine comes alive. The loop:
//   1. Encode the prompt into token IDs
//   2. Prepend BOS token
//   3. Run forward pass for each prompt token (fills KV cache)
//   4. Sample next token from the logits
//   5. Feed that token back in, run forward again
//   6. Repeat until EOS or max length
//   7. Print each generated token as it appears
//
// Prompt processing (steps 2-3) is called "prefill" — we're not
// generating yet, just letting the model "read" the prompt.
//
// Generation (steps 4-6) is called "decode" — one new token per
// forward pass.

// max_tokens is set 10 for testing, 1 token takes 30 seconds to generate.
static void generate(GGUFModel& model, RunState& state,
                     const std::string& prompt,
                     int max_tokens = 10,
                     float temperature = 0.7f,
                     float top_p = 0.9f) {

    const auto& cfg = model.config();
    const auto& tok = model.tokenizer();
    int n_vocab = static_cast<int>(cfg.n_vocab);

    // ---- Step 1: Encode the prompt ----
    std::vector<uint32_t> prompt_tokens = tok.encode(prompt);

    // Prepend BOS token (the model expects this at the start)
    prompt_tokens.insert(prompt_tokens.begin(), tok.bos_token());

    std::cout << "Prompt tokens: " << prompt_tokens.size() << " [";
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        std::cout << prompt_tokens[i];
        if (i + 1 < prompt_tokens.size()) std::cout << ", ";
    }
    std::cout << "]\n\n";

    // ---- Step 2: Prefill — process prompt tokens ----
    // Run forward pass for each prompt token to fill KV cache.
    // We only care about the logits after the LAST prompt token.
    int pos = 0;
    int next_token = 0;

    // Let the model read the prompt
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        forward(model, state, static_cast<int>(prompt_tokens[i]), pos);
        pos++;
    }

    // Sample first generated token from logits after prefill
    if (temperature <= 0.0f) {
        next_token = sample_argmax(state.logits, n_vocab);
    } else {
        next_token = sample_top_p(state.logits, n_vocab, temperature, top_p);
    }

    // Print the first generated token
    // flush() forces the output to appear on screen right away instead of waiting in a buffer. 
    // This is what creates the "streaming" effect where you see words appear one by one.
    std::string token_str = tok.decode_stripped({static_cast<uint32_t>(next_token)});
    std::cout << token_str;
    std::cout.flush();  // Print immediately, don't buffer

    // ---- Step 3: Generate — one token at a time ----
    for (int i = 1; i < max_tokens; i++) {
        forward(model, state, next_token, pos);
        pos++;

        // Sample next token
        if (temperature <= 0.0f) {
            next_token = sample_argmax(state.logits, n_vocab);
        } else {
            next_token = sample_top_p(state.logits, n_vocab, temperature, top_p);
        }

        // Check for end of sequence
        if (static_cast<uint32_t>(next_token) == tok.eos_token()) {
            break;
        }

        // Check context window limit
        if (pos >= static_cast<int>(cfg.n_ctx)) {
            std::cout << "\n[reached max context length]\n";
            break;
        }

        // Print the token
        token_str = tok.decode_stripped({static_cast<uint32_t>(next_token)});
        std::cout << token_str;
        std::cout.flush();
    }

    std::cout << "\n";
}

// -------------------- main --------------------

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage:\n"
                      << "  ./engine <model.gguf> \"prompt text\"\n"
                      << "  ./engine <model.gguf>                    (default prompt)\n"
                      << "  ./engine <model.gguf> dump <tensor> [n]  (dump tensor)\n";
            return 1;
        }

        std::string path = argv[1];
        GGUFModel model(path);

        std::cout << "Loaded " << model.config().architecture
                  << " (" << model.config().n_layers << " layers, "
                  << model.config().n_embd << " dim, "
                  << model.config().n_vocab << " vocab)\n";

        // Handle dump mode (keep existing functionality)
        if (argc >= 4 && std::string(argv[2]) == "dump") {
            std::string name = argv[3];
            std::size_t n = 10;
            if (argc >= 5) {
                n = static_cast<std::size_t>(std::strtoull(argv[4], nullptr, 10));
                if (n == 0) n = 10;
            }
            model.dump_tensor(name, n);
            return 0;
        }

        // Allocate runtime state
        RunState state;
        state.allocate(model.config());

        // Get prompt from command line or use default
        std::string prompt;
        if (argc >= 3) {
            prompt = argv[2];
        } else {
            // TinyLlama chat template format
            prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is the meaning of life?</s>\n<|assistant|>\n";
        }

        std::cout << "Prompt: \"" << prompt << "\"\n";
        std::cout << "Generating (this will be slow with naive matmul)...\n\n";

        generate(model, state, prompt);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}