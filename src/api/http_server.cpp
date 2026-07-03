#include "api/http_server.h"
#include "pipeline/rag_pipeline.h"
#include "observability/metrics.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace rag_engine {

// ============================================================================
// RequestHandler Implementation
// ============================================================================

RequestHandler::RequestHandler(std::shared_ptr<RAGPipeline> pipeline)
    : pipeline_(std::move(pipeline)) {}

RequestHandler::~RequestHandler() = default;

std::string RequestHandler::parse_query_id(const std::string& body) {
    // Simple JSON parsing for query_id
    size_t id_pos = body.find("\"query_id\"");
    if (id_pos != std::string::npos) {
        size_t start = body.find(':', id_pos);
        size_t quote_start = body.find('"', start);
        size_t quote_end = body.find('"', quote_start + 1);
        if (quote_start != std::string::npos && quote_end != std::string::npos) {
            return body.substr(quote_start + 1, quote_end - quote_start - 1);
        }
    }
    // Generate one if not present
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return "q_" + std::to_string(now);
}

void RequestHandler::handle_query(uv_stream_t* client, const std::string& body) {
    auto start = std::chrono::steady_clock::now();

    // Parse request
    std::string query_text;
    int32_t top_k = 5;

    // Simple JSON parsing
    size_t text_pos = body.find("\"query_text\"");
    if (text_pos != std::string::npos) {
        size_t start = body.find(':', text_pos);
        size_t quote_start = body.find('"', start);
        size_t quote_end = body.find('"', quote_start + 1);
        if (quote_start != std::string::npos && quote_end != std::string::npos) {
            query_text = body.substr(quote_start + 1, quote_end - quote_start - 1);
        }
    }

    size_t topk_pos = body.find("\"top_k\"");
    if (topk_pos != std::string::npos) {
        size_t start = body.find(':', topk_pos);
        top_k = std::stoi(body.substr(start + 1));
    }

    if (query_text.empty()) {
        send_error(client, 400, "query_text is required");
        return;
    }

    if (!pipeline_->is_ready()) {
        send_error(client, 503, "index not ready");
        return;
    }

    std::string query_id = parse_query_id(body);
    QueryRequest request(query_id, query_text, top_k);

    // Execute query
    auto results = pipeline_->query(request);

    // Calculate timing
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    // Record metrics
    MetricsCollector::instance().observe_histogram(metrics::QUERY_DURATION_SECONDS,
                                                   duration / 1'000'000.0);
    MetricsCollector::instance().increment_counter(metrics::QUERIES_TOTAL);
    MetricsCollector::instance().set_gauge(metrics::LAST_RESULT_COUNT, results.size());

    // Build response JSON
    std::ostringstream response;
    response << "{\"query_id\":\"" << query_id << "\",";
    response << "\"timing\":{\"total_us\":" << duration << "},";
    response << "\"results\":[";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        response << "{";
        response << "\"chunk_id\":\"" << r.chunk_id << "\",";
        response << "\"text\":\"" << r.text << "\",";
        response << "\"source\":\"" << r.source << "\",";
        response << "\"score\":" << std::fixed << std::setprecision(4) << r.similarity_score;
        response << "}";
        if (i < results.size() - 1) response << ",";
    }

    response << "]}";
    send_json(client, 200, response.str());
}

void RequestHandler::handle_index(uv_stream_t* client, const std::string& body) {
    std::string corpus_path;

    // Parse corpus_path from JSON
    size_t path_pos = body.find("\"corpus_path\"");
    if (path_pos != std::string::npos) {
        size_t start = body.find(':', path_pos);
        size_t quote_start = body.find('"', start);
        size_t quote_end = body.find('"', quote_start + 1);
        if (quote_start != std::string::npos && quote_end != std::string::npos) {
            corpus_path = body.substr(quote_start + 1, quote_end - quote_start - 1);
        }
    }

    bool rebuild = body.find("\"rebuild\":true") != std::string::npos;

    if (corpus_path.empty()) {
        send_error(client, 400, "corpus_path is required");
        return;
    }

    try {
        pipeline_->load_corpus(corpus_path);

        std::ostringstream response;
        response << "{\"success\":true,\"num_indexed\":" << pipeline_->get_batcher_stats().total_queries_submitted << "}";
        send_json(client, 200, response.str());
    } catch (const std::exception& e) {
        send_error(client, 500, e.what());
    }
}

void RequestHandler::handle_health(uv_stream_t* client) {
    std::ostringstream response;
    response << "{\"status\":\"" << (pipeline_->is_ready() ? "ok" : "loading") << "\"}";
    send_json(client, 200, response.str());
}

void RequestHandler::handle_metrics(uv_stream_t* client) {
    std::string metrics = MetricsCollector::instance().export_prometheus_format();

    std::ostringstream response;
    response << "HTTP/1.1 200 OK\r\n";
    response << "Content-Type: text/plain; version=0.0.4\r\n";
    response << "Content-Length: " << metrics.size() << "\r\n";
    response << "\r\n";
    response << metrics;

    // Send as raw response
    uv_buf_t buf = uv_buf_init(
        const_cast<char*>(response.str().c_str()),
        static_cast<unsigned int>(response.str().size())
    );

    uv_write_t* req = new uv_write_t;
    req->data = client;

    uv_write(req, client, &buf, 1, [](uv_write_t* req, int status) {
        delete req;
    });
}

void RequestHandler::send_json(uv_stream_t* client, int status, const std::string& json) {
    std::ostringstream response;
    response << "HTTP/1.1 " << status << " OK\r\n";
    response << "Content-Type: application/json\r\n";
    response << "Content-Length: " << json.size() << "\r\n";
    response << "Connection: close\r\n";
    response << "\r\n";
    response << json;

    uv_buf_t buf = uv_buf_init(
        const_cast<char*>(response.str().c_str()),
        static_cast<unsigned int>(response.str().size())
    );

    uv_write_t* req = new uv_write_t;
    req->data = client;

    uv_write(req, client, &buf, 1, [](uv_write_t* req, int status) {
        if (status < 0) {
            SPDLOG_ERROR("Write error: {}", uv_strerror(status));
        }
        delete req;
    });
}

void RequestHandler::send_error(uv_stream_t* client, int status, const std::string& message) {
    std::ostringstream body;
    body << "{\"error\":\"" << message << "\"}";

    send_json(client, status, body.str());
}

// ============================================================================
// HTTPServer Implementation
// ============================================================================

HTTPServer::HTTPServer(int port, std::shared_ptr<RequestHandler> handler)
    : port_(port), handler_(std::move(handler)) {
    loop_ = new uv_loop_t;
    uv_loop_init(loop_);
}

HTTPServer::~HTTPServer() {
    stop();
    uv_loop_close(loop_);
    delete loop_;
}

bool HTTPServer::start() {
    struct sockaddr_in addr;
    uv_ip4_addr("0.0.0.0", port_, &addr);

    uv_tcp_init(loop_, &server_);
    server_.data = this;

    if (uv_tcp_bind(&server_, reinterpret_cast<const struct sockaddr*>(&addr), 0) != 0) {
        SPDLOG_ERROR("Failed to bind to port {}", port_);
        return false;
    }

    if (uv_listen(reinterpret_cast<uv_stream_t*>(&server_), 128, on_connection) != 0) {
        SPDLOG_ERROR("Failed to listen on port {}", port_);
        return false;
    }

    running_ = true;
    SPDLOG_INFO("HTTP server started on port {}", port_);

    uv_run(loop_, UV_RUN_DEFAULT);
    return true;
}

void HTTPServer::stop() {
    if (!running_) return;

    running_ = false;
    uv_close(reinterpret_cast<uv_handle_t*>(&server_), nullptr);
    uv_stop(loop_);
}

void HTTPServer::on_connection(uv_stream_t* server, int status) {
    if (status < 0) {
        SPDLOG_ERROR("Connection error: {}", uv_strerror(status));
        return;
    }

    uv_tcp_t* client = new uv_tcp_t;
    uv_tcp_init(server->loop, client);
    client->data = server->data;  // Pass handler reference

    if (uv_accept(server, reinterpret_cast<uv_stream_t*>(client)) == 0) {
        uv_read_start(reinterpret_cast<uv_stream_t*>(client), alloc_buffer, on_read);
    } else {
        uv_close(reinterpret_cast<uv_handle_t*>(client), nullptr);
        delete client;
    }
}

void HTTPServer::alloc_buffer(uv_handle_t* handle, size_t suggested_size, uv_buf_t* buf) {
    buf->base = new char[suggested_size];
    buf->len = static_cast<unsigned int>(suggested_size);
}

void HTTPServer::on_read(uv_stream_t* client, ssize_t nread, const uv_buf_t* buf) {
    if (nread < 0) {
        if (nread != UV_EOF) {
            SPDLOG_ERROR("Read error: {}", uv_strerror(static_cast<int>(nread)));
        }
        uv_close(reinterpret_cast<uv_handle_t*>(client), nullptr);
        delete[] buf->base;
        return;
    }

    if (nread == 0) {
        delete[] buf->base;
        return;
    }

    std::string request(buf->base, static_cast<size_t>(nread));
    delete[] buf->base;

    // Parse HTTP request
    std::istringstream iss(request);
    std::string method, path, version;
    iss >> method >> path >> version;

    // Read body
    std::string body;
    size_t body_start = request.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        body = request.substr(body_start + 4);
    }

    HTTPServer* server = reinterpret_cast<HTTPServer*>(client->data);
    server->handle_request(client, method, path, body);
}

void HTTPServer::handle_request(uv_stream_t* client, const std::string& method,
                                const std::string& path, const std::string& body) {
    if (method == "POST" && path == "/query") {
        handler_->handle_query(client, body);
    } else if (method == "POST" && path == "/index") {
        handler_->handle_index(client, body);
    } else if (method == "GET" && path == "/health") {
        handler_->handle_health(client);
    } else if (method == "GET" && path == "/metrics") {
        handler_->handle_metrics(client);
    } else {
        std::ostringstream response;
        response << "HTTP/1.1 404 Not Found\r\n";
        response << "Content-Length: 0\r\n";
        response << "Connection: close\r\n\r\n";

        uv_buf_t buf = uv_buf_init(
            const_cast<char*>(response.str().c_str()),
            static_cast<unsigned int>(response.str().size())
        );
        uv_write_t* req = new uv_write_t;
        uv_write(req, client, &buf, 1, [](uv_write_t* req, int status) { delete req; });
    }
}

void HTTPServer::on_shutdown(uv_shutdown_t* req, int status) {
    uv_close(req->handle, nullptr);
    delete req;
}

void HTTPServer::after_write(uv_write_t* req, int status) {
    if (status < 0) {
        SPDLOG_ERROR("Write error: {}", uv_strerror(status));
    }
    delete req;
}

}  // namespace rag_engine
