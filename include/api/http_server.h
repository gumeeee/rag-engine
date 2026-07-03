#pragma once

#include <uv.h>
#include <memory>
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>

namespace rag_engine {

class RAGPipeline;

/**
 * HTTP request handler that processes incoming requests.
 */
class RequestHandler {
public:
    explicit RequestHandler(std::shared_ptr<RAGPipeline> pipeline);
    ~RequestHandler();

    void handle_query(uv_stream_t* client, const std::string& body);
    void handle_index(uv_stream_t* client, const std::string& body);
    void handle_health(uv_stream_t* client);
    void handle_metrics(uv_stream_t* client);

private:
    void send_json(uv_stream_t* client, int status, const std::string& json);
    void send_error(uv_stream_t* client, int status, const std::string& message);
    std::string parse_query_id(const std::string& body);

    std::shared_ptr<RAGPipeline> pipeline_;
};

/**
 * HTTP server using libuv for async, non-blocking I/O.
 */
class HTTPServer {
public:
    HTTPServer(int port, std::shared_ptr<RequestHandler> handler);
    ~HTTPServer();

    bool start();
    void stop();

private:
    static void on_connection(uv_stream_t* server, int status);
    static void alloc_buffer(uv_handle_t* handle, size_t suggested_size, uv_buf_t* buf);
    static void on_read(uv_stream_t* client, ssize_t nread, const uv_buf_t* buf);
    static void on_shutdown(uv_shutdown_t* req, int status);
    static void after_write(uv_write_t* req, int status);

    void handle_request(uv_stream_t* client, const std::string& method,
                       const std::string& path, const std::string& body);

    int port_;
    std::shared_ptr<RequestHandler> handler_;
    uv_loop_t* loop_;
    uv_tcp_t server_;
    bool running_{false};
};

}  // namespace rag_engine
