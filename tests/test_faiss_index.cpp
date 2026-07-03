#include <gtest/gtest.h>
#include "search/faiss_index.h"
#include "common/config.h"
#include "common/types.h"

using namespace rag_engine;

class FAISSIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        ConfigManager::RAGConfig::SearchConfig config;
        config.ef_search = 64;
        config.m_parameter = 32;
        config.ef_construction = 40;
        index = std::make_shared<FAISSIndex>(config);
    }

    std::shared_ptr<FAISSIndex> index;
};

TEST_F(FAISSIndexTest, NewIndexIsEmpty) {
    EXPECT_EQ(index->size(), 0);
    EXPECT_EQ(index->dimension(), 384);
    EXPECT_FALSE(index->is_loaded());
}

TEST_F(FAISSIndexTest, SearchOnEmptyIndexReturnsEmpty) {
    Embedding query(384);
    auto results = index->search(query, 10);

    EXPECT_TRUE(results.empty());
}

TEST_F(FAISSIndexTest, IndexConfiguration) {
    ConfigManager::RAGConfig::SearchConfig config;
    config.ef_search = 128;  // Higher value for testing
    config.m_parameter = 16;
    config.ef_construction = 32;

    auto test_index = std::make_shared<FAISSIndex>(config);
    EXPECT_EQ(test_index->dimension(), 384);
}
