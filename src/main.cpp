#include <cstdint>
#include <cstring>
#include <cerrno>
#include <cstdlib>
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
#include <cmath>
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
// “RAII” means cleanup happens automatically.

struct MMapFile {
    //file descriptor (Linux/macOS “file handle”). -1 means “not opened”.
    int fd = -1;
    //base: pointer to the start of the mapped file in memory.
    std::uint8_t* base = nullptr;
    //size: file size in bytes
    std::uint64_t size = 0;

    explicit MMapFile(const std::string& path) {
        //1) Open the file (read-only)
        
        //path.c_str() converts a C++ std::string into a C-style string (const char*)
        //O_RDONLY = open for reading only.
        //if it fails → throw error.
        //fd is just an ID number that the operating system uses to look up the real file information inside the kernel.
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error(sys_err("open"));

        //2) Get file size

        /*
        fstat fills st with info about the file (including size).
        If error or size 0 → close file and throw.
        */
        // “Using this file descriptor fd, fill st with info about the file.”
        //stat is a struct the OS uses to return file info (size, permissions, etc.).
        //“Hey OS, look up the file associated with descriptor fd and copy its info into st.”
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
        MAP_PRIVATE = changes (if any) won’t write back to disk
        fd = which file
        0 = AKA Offset into the file (0 means start at beginning).
        */
        void* p = ::mmap(nullptr, static_cast<size_t>(size), PROT_READ, MAP_PRIVATE, fd, 0);
        
        //If mapping fails → close fd and throw.
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


    //“This object cannot be copied.”
    //= delete means “this operation is forbidden”
    //“The copy constructor does not exist.”
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

    // const here means calling this function will NOT modify the object.”
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

// -------------------- Helper: Embedding Lookup --------------------
//
// The embedding table is a big matrix: [32000 × 2048]
// Each row is one token's "meaning" as a vector of 2048 numbers.
// This function copies row `token` into the output buffer,
// converting from F16 to F32 if needed.
//
// This is NOT a matmul — it's just a table lookup. No math,
// just "give me the 2048 numbers for token #15043."

static void embed_token(float* out, const GGUFModel& model, int token) {
    const auto& t = model.tensor_info("token_embd.weight");
    const std::uint8_t* data = model.tensor_bytes(t);
    int n_embd = static_cast<int>(model.config().n_embd);

    if (t.type == GGML_TYPE_F16) {
        // F16: each weight is 2 bytes, need to convert to F32
        const std::uint16_t* emb = reinterpret_cast<const std::uint16_t*>(data);
        const std::uint16_t* row = emb + static_cast<std::size_t>(token) * n_embd;
        for (int i = 0; i < n_embd; i++) {
            out[i] = fp16_to_f32(row[i]);
        }
    } else if (t.type == GGML_TYPE_F32) {
        // F32: each weight is 4 bytes, can copy directly
        const float* emb = reinterpret_cast<const float*>(data);
        const float* row = emb + static_cast<std::size_t>(token) * n_embd;
        for (int i = 0; i < n_embd; i++) {
            out[i] = row[i];
        }
    } else {
        throw std::runtime_error("Unsupported embedding type");
    }
}


// -------------------- Helper: Get Norm Weights --------------------
//
// RMSNorm weights are small 1D vectors (2048 floats = 8KB).
// They're always stored as F32 even in F16 models because:
//   - They're tiny (no memory savings from quantizing)
//   - Precision matters for normalization stability

static const float* get_norm_weight(const GGUFModel& model, const std::string& name) {
    const auto& t = model.tensor_info(name);
    if (t.type != GGML_TYPE_F32) {
        throw std::runtime_error("Expected F32 norm weight: " + name);
    }
    return reinterpret_cast<const float*>(model.tensor_bytes(t));
}


// -------------------- Helper: Get Weight Pointer + Type --------------------
//
// For matmul, we need both the raw pointer and the type
// (so matmul knows whether to use F16 or F32 path).

struct WeightRef {
    const void* data;
    ggml_type type;
};

static WeightRef get_weight(const GGUFModel& model, const std::string& name) {
    const auto& t = model.tensor_info(name);
    return { model.tensor_bytes(t), t.type };
}


// -------------------- The Forward Pass --------------------
//
// This is where everything comes together. One call to this function
// processes a single token through the entire model.
//
// Input:  a token ID (like 15043 for "Hello") and its position
// Output: state.logits filled with 32000 scores (one per vocab word)
//
// The flow:
//
//   Token ID ─── embed_token ───► x [2048]
//                                  │
//        ┌─────────────────────────┤  (repeat 22 times)
//        │                         ▼
//        │              rmsnorm ──► xb [2048]
//        │                         │
//        │              matmul(W_q, xb) ──► q  [2048]
//        │              matmul(W_k, xb) ──► k  [256]
//        │              matmul(W_v, xb) ──► v  [256]
//        │                         │
//        │              rope(q, k, pos)
//        │                         │
//        │              attention ──► xb2 [2048]
//        │                         │
//        │              matmul(W_o, xb2) ──► xb [2048]
//        │                         │
//        │              x = x + xb  (residual)
//        │                         │
//        │              rmsnorm ──► xb [2048]
//        │                         │
//        │              ffn_swiglu ──► xb2 [2048]
//        │                         │
//        │              x = x + xb2 (residual)
//        │                         │
//        └─────────────────────────┘
//                                  │
//                       rmsnorm ──► x [2048]  (final norm, in-place)
//                                  │
//                       matmul(W_out, x) ──► logits [32000]
//

static void forward(GGUFModel& model, RunState& state, int token, int pos) {
    const auto& cfg = model.config();
    int n_embd  = static_cast<int>(cfg.n_embd);
    int n_layers = static_cast<int>(cfg.n_layers);
    int n_ff    = static_cast<int>(cfg.n_ff);
    int kv_dim  = static_cast<int>(cfg.n_head_kv * cfg.head_dim());


    // ============================================================
    // Step 1: Embedding Lookup
    // ============================================================
    // Convert token ID → 2048-dimensional vector.
    // "Hello" (ID 15043) → [0.023, -0.841, 0.127, ..., 0.445]
    //
    // This is the starting point. These 2048 numbers are the model's
    // initial "understanding" of this word, before any context.

    embed_token(state.x, model, token);


    // ============================================================
    // Step 2: Transformer Layers (the main loop)
    // ============================================================
    // Each layer refines the representation.
    // After 22 layers of refinement, x contains enough information
    // to predict the next word.

    for (int l = 0; l < n_layers; l++) {

        // Build tensor name prefix for this layer: "blk.0.", "blk.1.", etc.
        std::string prefix = "blk." + std::to_string(l) + ".";


        // ---- ATTENTION HALF ----

        // RMSNorm: normalize x before attention.
        // Result goes to xb (we need original x for the residual later).
        const float* attn_norm_w = get_norm_weight(model, prefix + "attn_norm.weight");
        rmsnorm(state.xb, state.x, attn_norm_w, n_embd, cfg.rms_norm_eps);

        // Q, K, V projections: multiply normalized input by weight matrices.
        // This creates the query ("what am I looking for?"),
        // key ("what do I contain?"), and value ("here's my content")
        // for this token.
        auto wq = get_weight(model, prefix + "attn_q.weight");
        auto wk = get_weight(model, prefix + "attn_k.weight");
        auto wv = get_weight(model, prefix + "attn_v.weight");

        matmul(state.q, wq.data, state.xb, n_embd, n_embd, wq.type);
        matmul(state.k, wk.data, state.xb, kv_dim,  n_embd, wk.type);
        matmul(state.v, wv.data, state.xb, kv_dim,  n_embd, wv.type);

        // RoPE: rotate Q and K to encode position information.
        // Without this, "dog bites man" and "man bites dog" look the same.
        rope(state.q, state.k, pos,
             static_cast<int>(cfg.head_dim()),
             static_cast<int>(cfg.n_head),
             static_cast<int>(cfg.n_head_kv),
             cfg.rope_freq_base);

        // Attention: look back at all previous tokens, compute relevance
        // scores, and blend their values. Result → xb2
        attention(state.xb2, state.q, state.k, state.v,
                  state.key_cache, state.value_cache, state.att,
                  l, pos, cfg);

        // Output projection: convert attention output back to hidden dimension.
        // Result → xb (we're done with the old xb from rmsnorm)
        auto wo = get_weight(model, prefix + "attn_output.weight");
        matmul(state.xb, wo.data, state.xb2, n_embd, n_embd, wo.type);

        // Residual connection: add attention's contribution to original signal.
        // x = x + attention_output
        // The original x passes through unchanged; attention just adds to it.
        vec_add(state.x, state.xb, n_embd);


        // ---- FFN HALF ----

        // RMSNorm: normalize x before FFN.
        const float* ffn_norm_w = get_norm_weight(model, prefix + "ffn_norm.weight");
        rmsnorm(state.xb, state.x, ffn_norm_w, n_embd, cfg.rms_norm_eps);

        // SwiGLU FFN: expand to 5632 dims, process, shrink back to 2048.
        // This is the "thinking" step. Result → xb2
        auto wgate = get_weight(model, prefix + "ffn_gate.weight");
        auto wup   = get_weight(model, prefix + "ffn_up.weight");
        auto wdown = get_weight(model, prefix + "ffn_down.weight");

        ffn_swiglu(state.xb2, state.xb,
                   state.hb, state.hb2,
                   wgate.data, wup.data, wdown.data,
                   n_ff, n_embd, wgate.type);

        // Residual connection: add FFN's contribution.
        // x = x + ffn_output
        vec_add(state.x, state.xb2, n_embd);
    }


    // ============================================================
    // Step 3: Final RMSNorm
    // ============================================================
    // One last normalization before projecting to vocabulary.
    // This is in-place: x gets normalized directly (no need to
    // preserve the old x since there are no more residual connections).

    const float* final_norm_w = get_norm_weight(model, "output_norm.weight");
    rmsnorm(state.x, state.x, final_norm_w, n_embd, cfg.rms_norm_eps);


    // ============================================================
    // Step 4: Project to Vocabulary → Logits
    // ============================================================
    // Multiply the 2048-dim hidden state by a [32000 × 2048] matrix
    // to get one score per vocabulary word.
    //
    // TinyLlama uses "tied embeddings": the output matrix is the SAME
    // as the input embedding matrix. Instead of storing two 32000×2048
    // matrices, it reuses one. We check if "output.weight" exists;
    // if not, we fall back to "token_embd.weight".
    //
    // After this, state.logits[i] = how likely token i is to be next.
    // High logit = likely. Low logit = unlikely.

    WeightRef w_out;
    bool found_output = false;
    try {
        w_out = get_weight(model, "output.weight");
        found_output = true;
    } catch (...) {
        // Tied embeddings: reuse token embedding weights
    }
    if (!found_output) {
        w_out = get_weight(model, "token_embd.weight");
    }

    matmul(state.logits, w_out.data, state.x,
           static_cast<int>(cfg.n_vocab), n_embd, w_out.type);

    // state.logits now contains 32000 scores.
    // To get the next token: apply softmax, then sample.
}



// -------------------- main --------------------

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage:\n"
                      << "  ./engine <model.gguf>\n"
                      << "  ./engine <model.gguf> dump <tensor_name> [n]\n";
            return 1;
        }

        std::string path = argv[1];
        GGUFModel model(path);


        std::cout << "Loaded GGUF OK\n"
          << "  version: " << model.version() << "\n"
          << "  n_tensors: " << model.n_tensors() << "\n"
          << "  n_kv: " << model.n_kv() << "\n"
          << "  alignment: " << model.alignment() << "\n"
          << "  tensor_data_start: " << model.tensor_data_start() << "\n"
          << "  file_size: " << model.file_size() << "\n"
          << "\nModel Config:\n"
          << "  architecture: " << model.config().architecture << "\n"
          << "  n_layers: " << model.config().n_layers << "\n"
          << "  n_embd: " << model.config().n_embd << "\n"
          << "  n_head: " << model.config().n_head << "\n"
          << "  n_head_kv: " << model.config().n_head_kv << "\n"
          << "  n_ff: " << model.config().n_ff << "\n"
          << "  n_vocab: " << model.config().n_vocab << "\n"
          << "  n_ctx: " << model.config().n_ctx << "\n"
          << "  rope_dim: " << model.config().rope_dim << "\n"
          << "  head_dim: " << model.config().head_dim() << "\n"
          << "\nTokenizer:\n"
          << "  vocab_size: " << model.tokenizer().vocab_size() << "\n"
          << "  score_size: " << model.tokenizer().scores_size() << "\n"
          << "  unk_token: " << model.tokenizer().unk_token() << "\n"
          << "  bos_token: " << model.tokenizer().bos_token() << "\n"
          << "  eos_token: " << model.tokenizer().eos_token() << "\n";

        std::vector<std::uint32_t> test_tokens = {1, 2, 3};
        std::cout << "  decode({1,2,3}): \"" << model.tokenizer().decode_stripped(test_tokens) << "\"\n";

        std::string test_text = "Hello";
        std::vector<uint32_t> encoded = model.tokenizer().encode(test_text);
        std::cout << "  encode(\"Hello\"): [";
        for (size_t i = 0; i < encoded.size(); i++) {
            std::cout << encoded[i];
            if (i + 1 < encoded.size()) std::cout << ", ";
        }
        std::cout << "]\n";

        // Test round-trip
        std::string decoded = model.tokenizer().decode_stripped(encoded);
        std::cout << "  decode back: \"" << decoded << "\"\n";

        // Test various inputs
        std::vector<std::string> test_cases = {
            "Hello",
            " Hello",      // with leading space
            "Hello world",
            "The quick brown fox",
            "123",
            "don't",
        };

        std::cout << "\nTokenizer tests:\n";
        for (const auto& text : test_cases) {
            auto ids = model.tokenizer().encode(text);
            auto decoded = model.tokenizer().decode_stripped(ids);
            
            std::cout << "  input string: '" << text << "' → encoded values of string: [";
            for (size_t i = 0; i < ids.size(); i++) {
                std::cout << ids[i];
                if (i + 1 < ids.size()) std::cout << ", ";
            }
            std::cout << "] → decoded values of ids: '" << decoded << "'";
            
            // Check round-trip
            if (decoded == text) {
                std::cout << " ✓\n";
            } else {
                std::cout << " ✗ MISMATCH\n";
            }
        }

        // Forward pass smoke test
        RunState state;
        state.allocate(model.config());
        int bos = static_cast<int>(model.tokenizer().bos_token());
        forward(model, state, bos, 0);

        // Find top prediction
        float max_logit = state.logits[0];
        int max_id = 0;
        for (int i = 1; i < static_cast<int>(model.config().n_vocab); i++) {
            if (state.logits[i] > max_logit) {
                max_logit = state.logits[i];
                max_id = i;
            }
        }
        std::cout << "Top prediction after BOS: token " << max_id 
        << " (\"" << model.tokenizer().decode_stripped({static_cast<uint32_t>(max_id)}) << "\")\n";

        if (argc >= 4 && std::string(argv[2]) == "dump") {
            std::string name = argv[3];
            std::size_t n = 10;
            if (argc >= 5) {
                n = static_cast<std::size_t>(std::strtoull(argv[4], nullptr, 10));
                if (n == 0) n = 10;
            }
            std::cout << "\nDumping tensor\n";
            model.dump_tensor(name, n);
            std::cout << "\n";
        } else {
            // quick default sanity dumps
            std::cout << "\nDumping 2 tensors\n";
            model.dump_tensor("blk.0.attn_norm.weight", 10);
            std::cout << "\n";
            model.dump_tensor("token_embd.weight", 10);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}


