#include <gtest/gtest.h>
#include "embedding/cpu_embedding.h"
#include "embedding/embedding_service.h"
#include <future>

using namespace rag_engine;

class EmbeddingServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        service = std::make_shared<CPUEmbeddingService>(384);
    }

    std::shared_ptr<IEmbeddingService> service;
};

TEST_F(EmbeddingServiceTest, EncodeSingleText) {
    auto future = service->encode_async("Hello world");
    auto embedding = future.get();

    EXPECT_EQ(embedding.dimension, 384);
    EXPECT_FALSE(embedding.empty());
    EXPECT_EQ(embedding.size(), 384);
}

TEST_F(EmbeddingServiceTest, EncodeBatch) {
    std::vector<std::string> texts = {"Hello", "World", "Test"};
    auto future = service->encode_batch_async(texts);
    auto embeddings = future.get();

    EXPECT_EQ(embeddings.size(), 3);
    for (const auto& emb : embeddings) {
        EXPECT_EQ(emb.dimension, 384);
    }
}

TEST_F(EmbeddingServiceTest, SameTextProducesSameEmbedding) {
    auto future1 = service->encode_async("Same text");
    auto future2 = service->encode_async("Same text");

    auto emb1 = future1.get();
    auto emb2 = future2.get();

    // Same text should produce same embedding (deterministic)
    EXPECT_EQ(emb1.data, emb2.data);
}

TEST_F(EmbeddingServiceTest, DifferentTextsProduceDifferentEmbeddings) {
    auto future1 = service->encode_async("Text A");
    auto future2 = service->encode_async("Text B");

    auto emb1 = future1.get();
    auto emb2 = future2.get();

    // Different texts should produce different embeddings
    bool all_same = true;
    for (size_t i = 0; i < emb1.data.size() && all_same; ++i) {
        if (emb1.data[i] != emb2.data[i]) {
            all_same = false;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST_F(EmbeddingServiceTest, UsesCPU) {
    EXPECT_FALSE(service->uses_gpu());
    EXPECT_EQ(service->name(), "CPU");
}
