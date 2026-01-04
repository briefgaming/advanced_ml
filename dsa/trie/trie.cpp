#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <memory>


class TrieNode {
public:
    std::vector<std::unique_ptr<TrieNode>> children;
    bool is_end;

    TrieNode() : is_end(false) {
        children.resize(26);
    }

    bool containsChar(char ch) const {
        return children[ch - 'a'] != nullptr;
    }

    TrieNode* getChar(char ch) const {
        return children[ch - 'a'].get();
    }

    void putChar(char ch) {
        children[ch - 'a'] = std::make_unique<TrieNode>();
    }

    bool isEnd() {
        return is_end;
    }

    void setEnd() {
        is_end = true;
    }
};


class Trie {
private:
    std::unique_ptr<TrieNode> root;

    TrieNode* searchPrefix(const std::string& word) {
        TrieNode* node = root.get();
        for (char w : word) {
            if (node->containsChar(w)) {
                node = node->getChar(w);
            } else {
                return nullptr;
            }
        }
        return node;
    }

public:
    Trie() : root(std::make_unique<TrieNode>()) {}

    void insert(const std::string& word) {
        TrieNode* node = root.get();
        for (char w : word) {
            if (!node->containsChar(w)) {
                node->putChar(w);
            }
            node = node->getChar(w);
        }
        node->setEnd();
    }

    bool search(const std::string& word) {
        TrieNode* node = searchPrefix(word);
        return node != nullptr && node->isEnd();
    }

    bool startsWith(const std::string& prefix) {
        TrieNode* node = searchPrefix(prefix);
        return node != nullptr;
    }
};


void assertTrue(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "Assertion failed" << message << '\n';
        std::exit(1);
    }
}

void run_performance_test() {
    Trie trie;

    std::cout << "Running Trie Test Suite..." << '\n';

    // 1. Basic Insertion and Search
    trie.insert("apple");
    assertTrue(trie.search("apple") == true, "Test 1 Failed: 'apple' should be found after insertion.");
    assertTrue(trie.search("app") == false, "Test 1 Failed: 'app' should not be found yet (only 'apple' exists).");

    // 2. startsWith Functionality
    assertTrue(trie.startsWith("app") == true, "Test 2 Failed: 'app' is a valid prefix of 'apple'.");
    assertTrue(trie.startsWith("apple") == true, "Test 2 Failed: A word is also a prefix of itself.");
    assertTrue(trie.startsWith("b") == false, "Test 2 Failed: 'b' is not in the trie.");

    // 3. Inserting a Prefix of an Existing Word
    trie.insert("app");
    assertTrue(trie.search("app") == true, "Test 3 Failed: 'app' should be found after explicit insertion.");
    assertTrue(trie.search("apple") == true, "Test 3 Failed: 'apple' should still exist.");

    // 4. Inserting a Word that Extends an Existing Word
    trie.insert("applepie");
    assertTrue(trie.search("applepie") == true, "Test 4 Failed: 'applepie' should be found.");
    assertTrue(trie.search("apple") == true, "Test 4 Failed: 'apple' should still be found.");
    assertTrue(trie.startsWith("applep") == true, "Test 4 Failed: 'applep' is a prefix of 'applepie'.");

    // 5. Branching Paths (Distinct Words)
    trie.insert("bat");
    trie.insert("ball");
    assertTrue(trie.search("bat") == true, "Test 5 Failed: 'bat' should be found.");
    assertTrue(trie.search("ball") == true, "Test 5 Failed: 'ball' should be found.");
    assertTrue(trie.startsWith("ba") == true, "Test 5 Failed: 'ba' is a prefix for both.");
    assertTrue(trie.search("ba") == false, "Test 5 Failed: 'ba' was never inserted as a whole word.");

    // 6. Edge Case: Empty String
    trie.insert("");
    assertTrue(trie.search("") == true, "Test 6 Failed: Empty string search should return True after insertion.");
    assertTrue(trie.startsWith("") == true, "Test 6 Failed: Empty string is a prefix of everything/root.");

    // 7. Edge Case: Single Character Word
    trie.insert("z");
    assertTrue(trie.search("z") == true, "Test 7 Failed: Single char 'z' should be found.");
    assertTrue(trie.startsWith("z") == true, "Test 7 Failed: 'z' should be a prefix.");

    // 8. Redundant Insertion
    trie.insert("cat");
    trie.insert("cat");
    assertTrue(trie.search("cat") == true, "Test 8 Failed: 'cat' should be found after duplicate insert.");

    // 9. Non-existent Prefix vs Non-existent Word
    assertTrue(trie.search("banana") == false, "Test 9 Failed: 'banana' was never inserted.");
    assertTrue(trie.startsWith("ban") == false, "Test 9 Failed: No word starts with 'ban'.");

    // 10. Long Word Stress Test
    std::string long_word(100, 'a'); // Creates a string of 100 'a's
    trie.insert(long_word);
    assertTrue(trie.search(long_word) == true, "Test 10 Failed: 100-char word should be found.");
    
    std::string half_long_word(50, 'a');
    assertTrue(trie.startsWith(half_long_word) == true, "Test 10 Failed: 50-char prefix should be valid.");
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0; i < 20000; i++) {
        run_performance_test();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;

    std::cout << "Tests passed. Avg time: " << (diff.count() / 20) << "ms\n";
    return 0;

}