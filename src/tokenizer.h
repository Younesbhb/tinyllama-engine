#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>



class Tokenizer{
    public:
        void load(std::vector<std::string>&& vocab, std::vector<float>&& scores);


        std::string decode(const std::vector<uint32_t>& tokens) const;
        std::string decode_stripped(const std::vector<uint32_t>& tokens) const;
        std::vector<uint32_t> encode(const std::string& text) const;

        // Special token IDs (filled during load)
        uint32_t bos_token() const;  // Beginning of sequence
        uint32_t eos_token() const;   // End of sequence
        uint32_t unk_token() const;   // Unknown token


        uint32_t vocab_size() const; 
        uint32_t scores_size() const;
        
    

    private:

        std::vector<std::string> vocab_;  // ID → String. Stores strings
        std::vector<float> scores_; 
        std::unordered_map<std::string, std::uint32_t> token_to_id_;  // String → ID

        uint32_t unk_id_ = 0;   // <unk>
        uint32_t bos_id_ = 1;   // <s>
        uint32_t eos_id_ = 2;   // </s>

};