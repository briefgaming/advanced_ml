// Define preprocessors
#include <iostream>
#include <unordered_map>


class RadixNode {
public:
    // Declare member variables/parameters
    std::unordered_map<char, std::unique_ptr<RadixNode>> children;
    std::string label;
    bool is_end;

    // Initialize constructor using member list
    RadixNode(const std::string& lbl = "", bool end = false)
        : label(lbl), is_end(end) {}

    static size_t getCommonPrefixLength(const std::string& word1, const std::string& word2) {
        size_t minLen = std::min(word1.length(), word2.length());
        for (size_t i=0; i<minLen; i++) {
            if (word1[i] != word2[i]) {
                return i;
            }
        }
        return minLen;
    }

    bool search(const std::string& word) {
        if (word.empty()) {
            return is_end;
        }

        char firstChar = word[0];
        auto iterator = children.find(firstChar);
        if (iterator == children.end()) {
            return false;
        }
        
        RadixNode* child = iterator->second.get();
        size_t prefixLen = RadixNode::getCommonPrefixLength(word, child->label);
        if (prefixLen == child->label.length()) {
            return child->search(word.substr(prefixLen));
        }
        return false;
    }
}