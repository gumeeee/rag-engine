#include "observability/metrics.h"
#include <sstream>
#include <iomanip>

namespace rag_engine {

MetricsCollector& MetricsCollector::instance() {
    static MetricsCollector instance;
    return instance;
}

void MetricsCollector::increment_counter(const std::string& name, double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& counter = counters_[name];
    counter.value += value;
    counter.last_update = std::chrono::steady_clock::now();
}

double MetricsCollector::get_counter(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = counters_.find(name);
    if (it != counters_.end()) {
        return it->second.value.load();
    }
    return 0.0;
}

void MetricsCollector::observe_histogram(const std::string& name, double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& histogram = histograms_[name];

    std::lock_guard<std::mutex> hist_lock(histogram.mutex);
    histogram.values.push_back(value);

    if (histogram.values.size() > histogram.max_size) {
        histogram.values.erase(histogram.values.begin());
    }
}

std::vector<double> MetricsCollector::get_histogram(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = histograms_.find(name);
    if (it != histograms_.end()) {
        std::lock_guard<std::mutex> hist_lock(it->second.mutex);
        return it->second.values;
    }
    return {};
}

void MetricsCollector::set_gauge(const std::string& name, double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    gauges_[name].value = value;
}

double MetricsCollector::get_gauge(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = gauges_.find(name);
    if (it != gauges_.end()) {
        return it->second.value.load();
    }
    return 0.0;
}

std::string MetricsCollector::export_prometheus_format() const {
    std::ostringstream oss;
    auto now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    std::lock_guard<std::mutex> lock(mutex_);

    // Export counters
    for (const auto& [name, counter] : counters_) {
        oss << "# TYPE " << name << " counter\n";
        oss << name << " " << std::fixed << std::setprecision(3) << counter.value.load()
            << " " << now << "\n";
    }

    // Export gauges
    for (const auto& [name, gauge] : gauges_) {
        oss << "# TYPE " << name << " gauge\n";
        oss << name << " " << std::fixed << std::setprecision(3) << gauge.value.load()
            << " " << now << "\n";
    }

    // Export histograms (as summaries for simplicity)
    for (const auto& [name, histogram] : histograms_) {
        std::vector<double> values;
        {
            std::lock_guard<std::mutex> hist_lock(histogram.mutex);
            values = histogram.values;
        }

        if (values.empty()) continue;

        // Calculate percentiles
        std::sort(values.begin(), values.end());

        double p50 = values[values.size() * 0.5];
        double p95 = values[values.size() * 0.95];
        double p99 = values[values.size() * 0.99];

        oss << "# TYPE " << name << "_p50 gauge\n";
        oss << name << "_p50 " << std::fixed << std::setprecision(6) << p50 << " " << now << "\n";
        oss << "# TYPE " << name << "_p95 gauge\n";
        oss << name << "_p95 " << std::fixed << std::setprecision(6) << p95 << " " << now << "\n";
        oss << "# TYPE " << name << "_p99 gauge\n";
        oss << name << "_p99 " << std::fixed << std::setprecision(6) << p99 << " " << now << "\n";
    }

    return oss.str();
}

void MetricsCollector::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_.clear();
    histograms_.clear();
    gauges_.clear();
}

}  // namespace rag_engine
