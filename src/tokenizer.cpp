#include "tokenizer.h"
#include <algorithm>  // for std::min
#include <stdexcept>
#include <cstdio>     // for snprintf, sscanf

// Initialize vocab_ and scores_ and token_to_id_
// scores_ is not used in our implementation
void Tokenizer::load(std::vector<std::string>&& vocab, std::vector<float>&& scores) {
    vocab_ = std::move(vocab);
    scores_ = std::move(scores);
    
    token_to_id_.reserve(vocab_.size());
    // Build reverse lookup
    for (size_t i = 0; i < vocab_.size(); i++) {
        token_to_id_[vocab_[i]] = static_cast<uint32_t>(i);
    }
    
    // Find special tokens positions (they might not be at fixed positions)
    auto find_token = [this](const std::string& s) -> std::uint32_t {
        auto it = token_to_id_.find(s);
        if(it == token_to_id_.end()){
            throw std::runtime_error("Required special token not found: " + s);
        }
        return it->second;
    };
    
    unk_id_ = find_token("<unk>");
    bos_id_ = find_token("<s>");
    eos_id_ = find_token("</s>");
}

// The decode function converts token IDs back into readable text. 
std::string Tokenizer::decode(const std::vector<std::uint32_t>& tokens) const {
    size_t total_size = 0;
    //First pass: calculate size
    for (auto id : tokens) {
        if (static_cast<size_t>(id) < vocab_.size()) {
            total_size += vocab_[id].size();
        }
    }

    std::string result;
    // Build raw result with reserved size
    result.reserve(total_size);
    for (auto id : tokens) {
        if (static_cast<size_t>(id) < vocab_.size()) {
            const std::string& tok = vocab_[id];
            // Check for byte tokens like <0x0A>, <0x0D>, etc.
            // These are 6 characters: '<' '0' 'x' H H '>'
            // Cheap check first
            if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>') {
                // Parse the two hex digits
                unsigned int byte_val = 0;
                // Expensive check only if needed
                if (std::sscanf(tok.c_str(), "<0x%02X>", &byte_val) == 1 && byte_val <= 255) {
                    // Convert the number to an actual character and append it. For byte_val = 10, static_cast<char>(10) is the newline character \n. 
                    // For byte_val = 32, it's a space. For byte_val = 13, it's carriage return \r.

                    // % 0 2 X
                    // │ │ │ │
                    // │ │ │ └── type: X = uppercase hexadecimal (A-F). Lowercase x would give a-f.
                    // │ │ └── width: at least 2 characters wide
                    // │ └── fill: pad with '0' instead of spaces
                    // └── start of specifier

                    result += static_cast<char>(byte_val);
                } else {
                    result += tok;  // couldn't parse, keep as-is
                }
            } else {
                result += tok;
            }
        }
    }
    
    // Post-process: replace ▁ with space
    std::string final_result;
    final_result.reserve(total_size);  // Same size (▁ shrinks to space)
    
    size_t i = 0;
    while (i < result.size()) {
        // Check for ▁ (3 bytes: 0xE2 0x96 0x81) to transform into space
        if (i + 2 < result.size() &&
            //The bytes 0xE2, 0x96, 0x81 are all > 127, so we need unsigned char for correct comparison.
            static_cast<unsigned char>(result[i]) == 0xE2 &&
            static_cast<unsigned char>(result[i + 1]) == 0x96 &&
            static_cast<unsigned char>(result[i + 2]) == 0x81) {
            final_result += ' ';
            i += 3;
        //Every character other than ▁
        } else {
            final_result += result[i];
            i += 1;
        }
    }
    
    return final_result;
}

std::string Tokenizer::decode_stripped(const std::vector<uint32_t>& tokens) const{
    std::string decoded = decode(tokens);
    if(!decoded.empty() && decoded[0] == ' '){
        return decoded.substr(1);
    }
    return decoded;
}


// Transform a string into an array of ids/tokens representing them
std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<uint32_t> tokens;
    
    // Preprocess: replace spaces with ▁ (SentencePiece convention)
    std::string processed;
    processed.reserve(text.size() +3);
    processed += "\xE2\x96\x81";

    
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == ' ') {
            // ▁ is Unicode U+2581, encoded as 3 bytes in UTF-8: 0xE2 0x96 0x81
            processed += "\xE2\x96\x81";
        } else {
            processed += text[i];
        }
    }
    
    size_t pos = 0;
    while (pos < processed.size()) {
        size_t best_len = 0;
        uint32_t best_id = unk_id_;
        
        size_t max_len = std::min(processed.size() - pos, size_t(64));
        
        for (size_t len = max_len; len >= 1; len--) {
            std::string candidate = processed.substr(pos, len);
            //Use string_view (C++17+): optional but not necessarily recommended
            //std::string_view candidate(processed.data() + pos, len);  // No allocation
            auto it = token_to_id_.find(candidate);
            if (it != token_to_id_.end()) {
                best_len = len;
                best_id = it->second;
                break;
            }
        }
        
        if (best_len > 0) {
            tokens.push_back(best_id);
            pos += best_len;
        } else {
            // Byte-level fallback: encode the unknown byte as <0xNN>
            // The vocabulary contains tokens like <0x0A> (newline), <0x0D> (carriage return), etc.
            // This handles any character our greedy matcher can't find a token for.
            unsigned char byte = static_cast<unsigned char>(processed[pos]);
            char hex[8];
            std::snprintf(hex, sizeof(hex), "<0x%02X>", byte);
            auto it = token_to_id_.find(std::string(hex));
            if (it != token_to_id_.end()) {
                tokens.push_back(it->second);
            } else {
                // Truly unknown byte — no byte token exists
                tokens.push_back(unk_id_);
            }
            pos += 1;
        }
    }
    
    return tokens;
}


//Can be used once vocab_ is filled at runtime
uint32_t Tokenizer::vocab_size() const {
    return static_cast<uint32_t>(vocab_.size());
}

//Can be used once scores_ is filled at runtime
uint32_t Tokenizer::scores_size() const{
    return static_cast<uint32_t>(scores_.size());
}


// Useful for Generation Loop
uint32_t Tokenizer::unk_token() const {
    return unk_id_;
}
// Useful for Generation Loop
uint32_t Tokenizer::bos_token() const {
    return bos_id_;
}
// Useful for Generation Loop
uint32_t Tokenizer::eos_token() const {
    return eos_id_;
}