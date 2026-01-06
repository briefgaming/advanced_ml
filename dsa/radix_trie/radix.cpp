#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cassert>
#include <chrono>


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

    bool deleteWord(const std::string& word) {
        if (word.empty()){
            if (!is_end) {
                return false;
            }
            is_end = false;
            return children.empty();
        }

        char firstChar = word[0];
        auto iterator = (children.find(firstChar));
        if (iterator == children.end()) {
            return false;
        }

        RadixNode* child = iterator->second.get();
        size_t prefixLen = getCommonPrefixLength(word, child->label);

        if (prefixLen != child->label.length()) {
            return false;
        }

        bool shouldPruneChild = child->deleteWord(word.substr(prefixLen));

        if (shouldPruneChild) {
            children.erase(firstChar);
        } else if (children.find(firstChar) != children.end()) {
            applyMerge(children[firstChar].get());
        }
        return children.empty() && !is_end;
    }

    void applyMerge(RadixNode* node) {
        if (!node->is_end && node->children.size() == 1) {
            // Access and iterator pointing to the first element in the map
            auto childIterator = node->children.begin();
            // Gets the raw pointer to the child node
            RadixNode* childNode = childIterator->second.get();

            std::string childLabel = childNode->label;
            bool childIsEnd = childNode->is_end;
            auto childChildren = std::move(childNode->children);

            node->label += childLabel;
            node->is_end = childIsEnd;
            node->children = std::move(childChildren);
        }

    }

    void findAllWithPrefix(const std::string& prefix, const std::string& visitedPath, std::vector<std::string>& results) {
        // Navigation phase
        if (prefix.length() > 0) {
            char firstChar = prefix[0];

            if (children.find(firstChar) == children.end()) {
                return;
            }

            RadixNode* child = children[firstChar].get();
            size_t prefixLen = getCommonPrefixLength(prefix, child->label);

            if (prefixLen == prefix.length()) {
                child->findAllWithPrefix("", visitedPath + child->label, results);
            } else if (prefixLen == child->label.length()) {
                // The prefix is large than the current node label so we recurse down the tree
                child->findAllWithPrefix(prefix.substr(prefixLen), visitedPath + child->label, results);
            }
            return;
        }

        // Collection phase
        if (is_end) {
            results.push_back(visitedPath);
        }

        for (auto& child : children) {
            child.second->findAllWithPrefix("", visitedPath + child.second->label, results);
        }
    }
};


class Radix {
private:
    std::unique_ptr<RadixNode> root;

public:
    Radix(): root(std::make_unique<RadixNode>()) {}

    void insert(const std::string& word) {
        RadixNode* current = root.get();
        std::string remaining = word;

        while (!remaining.empty()) {
            char firstChar = remaining[0];

            auto iterator = current->children.find(firstChar);
            if (iterator == current->children.end()) {
                current->children[firstChar] = std::make_unique<RadixNode>(remaining, true);
                return;
            }

            RadixNode* child = iterator->second.get();
            size_t prefixLen = RadixNode::getCommonPrefixLength(remaining, child->label);

            if (prefixLen == child->label.length()) {
                remaining = remaining.substr(prefixLen);
                current = child;
            } else {
                auto splitNode = std::make_unique<RadixNode>(child->label.substr(0, prefixLen));

                child->label = child->label.substr(prefixLen);
                splitNode->children[child->label[0]] = std::move(iterator->second);

                if (prefixLen == remaining.length()) {
                    splitNode->is_end = true;
                } else {
                    std::string remainingWord = remaining.substr(prefixLen);
                    splitNode->children[remainingWord[0]] = std::make_unique<RadixNode>(remainingWord, true);
                }
                current->children[firstChar] = std::move(splitNode);
                return;
            }
        }
        current->is_end = true;
    }

    bool search(const std::string& word) {
        return root->search(word);
    }

    bool deleteWord(const std::string& word) {
        return root->deleteWord(word);
    }

    std::vector<std::string> findAllWithPrefix(const std::string& prefix) {
        std::vector<std::string> results;
        root->findAllWithPrefix(prefix, "", results);
        return results;
    }
};


void assertTrue(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "Assertion failed: " << message << '\n';
        std::exit(1);
    }
}

void run_tests(bool quiet = false) {
    Radix tree;

    if (!quiet) std::cout << "--- Test 1: Basic Insert & Search ---\n";
    tree.insert("test");
    tree.insert("team");
    assertTrue(tree.search("test") == true, "Failed to find 'test'");
    assertTrue(tree.search("team") == true, "Failed to find 'team'");
    assertTrue(tree.search("tea") == false, "Found 'tea' which shouldn't exist yet");
    if (!quiet) std::cout << "✅ Basic Insert Passed\n";

    if (!quiet) std::cout << "\n--- Test 2: Split Logic (Case 3 & 4) ---\n";
    tree.insert("toast");
    tree.insert("toaster");
    tree.insert("toad");

    assertTrue(tree.search("toast") == true, "'toast' should exist");
    assertTrue(tree.search("toaster") == true, "'toaster' should exist");
    assertTrue(tree.search("toad") == true, "'toad' should exist");
    assertTrue(tree.search("toas") == false, "'toas' should not exist");
    if (!quiet) std::cout << "✅ Split Logic Passed\n";

    if (!quiet) std::cout << "\n--- Test 3: Prefix Search (Autocomplete) ---\n";
    auto results_t = tree.findAllWithPrefix("t");
    std::sort(results_t.begin(), results_t.end());
    std::vector<std::string> expected_t = {"team", "test", "toad", "toast", "toaster"};
    assertTrue(results_t == expected_t, "Prefix 't' results mismatch");

    auto results_toa = tree.findAllWithPrefix("toa");
    std::sort(results_toa.begin(), results_toa.end());
    std::vector<std::string> expected_toa = {"toad", "toast", "toaster"};
    assertTrue(results_toa == expected_toa, "Prefix 'toa' results mismatch");

    auto results_te = tree.findAllWithPrefix("te");
    std::sort(results_te.begin(), results_te.end());
    std::vector<std::string> expected_te = {"team", "test"};
    assertTrue(results_te == expected_te, "Prefix 'te' results mismatch");
    if (!quiet) std::cout << "✅ Prefix Search Passed\n";

    if (!quiet) std::cout << "\n--- Test 4: Deletion & Merging ---\n";
    tree.deleteWord("toaster");
    assertTrue(tree.search("toaster") == false, "'toaster' should be gone");
    assertTrue(tree.search("toast") == true, "'toast' should still exist");

    tree.deleteWord("toast");
    assertTrue(tree.search("toast") == false, "'toast' should be gone");
    assertTrue(tree.search("toad") == true, "'toad' should still exist");
    if (!quiet) std::cout << "✅ Deletion Passed\n";

    if (!quiet) std::cout << "\n--- Test 5: Edge Cases ---\n";
    tree.insert("a");
    tree.insert("ab");
    assertTrue(tree.search("a") == true, "'a' should exist");
    assertTrue(tree.search("ab") == true, "'ab' should exist");
    tree.deleteWord("a");
    assertTrue(tree.search("a") == false, "'a' should be gone");
    assertTrue(tree.search("ab") == true, "'ab' should still exist");
    if (!quiet) std::cout << "✅ Edge Cases Passed\n";

    if (!quiet) std::cout << "\n🎉 ALL TESTS PASSED!\n";
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    // Run once with output
    run_tests(false);

    // Run performance test quietly
    const int ITERATIONS = 20000;
    for (int i = 0; i < ITERATIONS; ++i) {
        run_tests(true);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;

    std::cout << "Tests passed. Total time: " << diff.count() << "ms\n";
    std::cout << "Average time per run: " << (diff.count() / ITERATIONS) << "ms\n";

    return 0;
}