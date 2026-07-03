#include <gtest/gtest.h>
#include "pipeline/query_batcher.h"
#include "embedding/cpu_embedding.h"
#include "common/config.h"
#include <future>

using namespace rag_engine;

class QueryBatcherTest : public ::testing::Test {
protected:
    void SetUp() override {
        ConfigManager::RAGConfig::BatcherConfig config;
        config.max_batch_size = 32;
        config.max_wait_ms = 10;
        config.num_dispatcher_threads = 1;

        auto embedding_service = std::make_shared<CPUEmbeddingService>(384);
        batcher = std::make_shared<QueryBatcher>(embedding_service, config);
    }

    void TearDown() override {
        batcher->shutdown();
    }

    std::shared_ptr<QueryBatcher> batcher;
};

TEST_F(QueryBatcherTest, SubmitSingleQuery) {
    auto future = batcher->submit({"test query"});
    auto embeddings = future.get();

    EXPECT_EQ(embeddings.size(), 1);
    EXPECT_EQ(embeddings[0].dimension, 384);
}

TEST_F(QueryBatcherTest, SubmitMultipleQueries) {
    std::vector<std::string> queries = {"query1", "query2", "query3"};
    auto future = batcher->submit(queries);
    auto embeddings = future.get();

    EXPECT_EQ(embeddings.size(), 3);
}

TEST_F(QueryBatcherTest, EmptyBatchReturnsEmpty) {
    auto future = batcher->submit({});
    auto embeddings = future.get();

    EXPECT_TRUE(embeddings.empty());
}

TEST_F(QueryBatcherTest, GetStatsAfterSubmission) {
    batcher->submit({"test1"});
    batcher->submit({"test2"});

    auto stats = batcher->get_stats();
    EXPECT_GE(stats.total_queries_submitted, 2);
}

TEST_F(QueryBatcherTest, FlushClearsPending) {
    batcher->submit({"test1"});
    batcher->submit({"test2"});

    batcher->flush();

    // After flush, all queries should be processed
    auto stats = batcher->get_stats();
    EXPECT_EQ(stats.total_batches_dispatched, 1);  // Should be in one batch
}
