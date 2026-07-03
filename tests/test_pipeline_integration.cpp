#include <gtest/gtest.h>
#include "pipeline/rag_pipeline.h"
#include "embedding/cpu_embedding.h"
#include "search/faiss_index.h"
#include "common/config.h"

using namespace rag_engine;

class PipelineIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create config
        ConfigManager::RAGConfig config;

        // Create embedding service
        auto embedding_service = std::make_shared<CPUEmbeddingService>(384);

        // Create batcher
        batcher = std::make_shared<QueryBatcher>(embedding_service, config.batcher);

        // Create vector store
        vector_store = std::make_shared<FAISSIndex>(config.search);

        // Create pipeline
        pipeline = std::make_shared<RAGPipeline>(batcher, vector_store, config);
    }

    void TearDown() override {
        batcher->shutdown();
    }

    std::shared_ptr<QueryBatcher> batcher;
    std::shared_ptr<IVectorStore> vector_store;
    std::shared_ptr<RAGPipeline> pipeline;
};

TEST_F(PipelineIntegrationTest, PipelineNotReadyByDefault) {
    EXPECT_FALSE(pipeline->is_ready());
}

TEST_F(PipelineIntegrationTest, QueryOnUnreadyPipelineReturnsEmpty) {
    QueryRequest request("test_id", "test query", 5);
    auto results = pipeline->query(request);

    EXPECT_TRUE(results.empty());
}

TEST_F(PipelineIntegrationTest, GetBatcherStats) {
    auto stats = pipeline->get_batcher_stats();

    EXPECT_EQ(stats.total_queries_submitted, 0);
    EXPECT_EQ(stats.total_batches_dispatched, 0);
}
