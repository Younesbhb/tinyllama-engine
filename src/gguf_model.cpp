#include "gguf_model.h"


#include <cerrno>
#include <stdexcept>
#include <limits>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


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
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q4_0: return "Q4_0";
        default: return "OTHER";
    }
}

// Compute the total byte size of a tensor given its element count and type.
// For F32/F16 this is simple: n_elems × bytes_per_element.
// For quantized types, weights are stored in blocks of 32 elements,
// so we compute: (n_elems / 32) × bytes_per_block.
static std::uint64_t tensor_byte_size(std::uint64_t n_elems, ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return n_elems * 4;
        case GGML_TYPE_F16: return n_elems * 2;
        case GGML_TYPE_Q8_0: {
            // 34 bytes per block of 32 elements
            if (n_elems % QK8_0 != 0)
                throw std::runtime_error("Q8_0 tensor size not a multiple of 32");
            return (n_elems / QK8_0) * sizeof(block_q8_0);
        }
        case GGML_TYPE_Q4_0: {
            // 18 bytes per block of 32 elements
            if (n_elems % QK4_0 != 0)
                throw std::runtime_error("Q4_0 tensor size not a multiple of 32");
            return (n_elems / QK4_0) * sizeof(block_q4_0);
        }
        default:
            throw std::runtime_error("Unsupported ggml_type for byte size calculation.");
    }
}



// -------------------- mmap RAII --------------------
// "RAII" means cleanup happens automatically.

MMapFile::MMapFile(const std::string& path) {
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

MMapFile::~MMapFile() {
    if (base && base != MAP_FAILED) {
        //unmap the memory (munmap)
        ::munmap(base, static_cast<size_t>(size));
    }
    if (fd >= 0) ::close(fd);
}



// -------------------- safe cursor over mapped bytes --------------------


Cursor::Cursor(const std::uint8_t* base, std::uint64_t size) : p(base), end(base + size), off(0) {}

void Cursor::need(std::uint64_t n) const {
    if (static_cast<std::uint64_t>(end - p) < n) {
        throw std::runtime_error("Unexpected EOF while parsing");
    }
}

const std::uint8_t* Cursor::take(std::uint64_t n) {
    //n is the nb of bytes
    need(n);
    const std::uint8_t* out = p;
    p += n;
    off += n;
    return out;
}

std::string Cursor::read_string() {
    std::uint64_t len = read_pod<std::uint64_t>();
    if (len > (1ull << 30)) throw std::runtime_error("Suspicious string length (>1GB)");
    const std::uint8_t* b = take(len);
    return std::string(reinterpret_cast<const char*>(b), static_cast<size_t>(len));
}

void Cursor::skip_string() {
    std::uint64_t len = read_pod<std::uint64_t>();
    //we can do that since len represents the length (nb of char == nb of bytes) in the string
    take(len);
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

// -------------------- Model class (owns mmap + parsed index) --------------------



GGUFModel::GGUFModel(const std::string& path)
    : mm_(path), cur_(mm_.base, mm_.size) {
    parse();
}

// const here means calling this function will NOT modify the object.
std::uint32_t GGUFModel::version() const { return version_; }
std::uint64_t GGUFModel::n_tensors() const { return n_tensors_; }
std::uint64_t GGUFModel::n_kv() const { return n_kv_; }
std::uint64_t GGUFModel::alignment() const { return alignment_; }
std::uint64_t GGUFModel::tensor_data_start() const { return tensor_data_start_; }
std::uint64_t GGUFModel::file_size() const { return mm_.size; }

const gguf_tensor_info_t& GGUFModel::tensor_info(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) throw std::runtime_error("Tensor not found: " + name);
    return tensors_.at(it->second);
}

const std::uint8_t* GGUFModel::tensor_bytes(const gguf_tensor_info_t& t) const {
    // abs_offset computed during parse + sanity checks
    return mm_.base + t.abs_offset;
}

const llama_config_t& GGUFModel::config() const{
    return config_;
}

const Tokenizer& GGUFModel::tokenizer() const {return tokenizer_;}




void GGUFModel::dump_tensor(const std::string& name, std::size_t n) const {
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

void GGUFModel::parse() {
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

        // compute byte_size (handles both per-element and block-based types)
        t.byte_size = tensor_byte_size(n, t.type);

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