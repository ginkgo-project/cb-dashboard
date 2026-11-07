// mpibench: the time of the MPI operations Ginkgo's distributed code uses, over
// a sweep of message sizes, on device buffers (GPU-aware MPI). The weekly
// distributed run measures it before the Ginkgo benchmarks, in the same
// allocation, so that a change in Ginkgo's times can be told apart from a
// change in MPI or the network. It doesn't use Ginkgo, so it doesn't change
// with Ginkgo's commits.
//
// Usage: mpibench [--max-allreduce-k K] [--max-alltoallv-k K] [--pingpong]
//                 [--max-pingpong-bytes B] [--ranks-per-node R] [--min-time S]
//
// Rank 0 prints a JSON array of cases shaped like Ginkgo's benchmark output,
// {"operation", "count", "bytes", "ranks", "mpi": {<operation>: {"completed",
// "time", "min_time", "repetitions"[, "error"]}}}. A repetition starts with a
// barrier and takes the slowest rank's time; "time" is the median of the
// repetitions, "min_time" the fastest. The ping-pong's times are one way (half
// a round trip).

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#if defined(MPIBENCH_CUDA)
#include <cuda_runtime.h>
#elif defined(MPIBENCH_HIP)
#include <hip/hip_runtime.h>
#endif


namespace {


[[noreturn]] void die(const std::string& message)
{
    std::fprintf(stderr, "mpibench: %s\n", message.c_str());
    MPI_Abort(MPI_COMM_WORLD, 2);
    std::abort();
}


// Buffers: device memory with CUDA or HIP, host memory otherwise.
#if defined(MPIBENCH_CUDA)
#define DEVICE_CHECK(call)                                               \
    do {                                                                 \
        auto err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                       \
            die(std::string(#call) + ": " + cudaGetErrorString(err_));   \
        }                                                                \
    } while (false)
void* device_alloc(std::size_t bytes)
{
    void* p = nullptr;
    DEVICE_CHECK(cudaMalloc(&p, std::max<std::size_t>(bytes, 8)));
    return p;
}
void device_free(void* p) { DEVICE_CHECK(cudaFree(p)); }
void to_device(void* dst, const void* src, std::size_t bytes)
{
    DEVICE_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}
void to_host(void* dst, const void* src, std::size_t bytes)
{
    DEVICE_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}
const char* buffers = "cuda";
#elif defined(MPIBENCH_HIP)
#define DEVICE_CHECK(call)                                              \
    do {                                                                \
        auto err_ = (call);                                             \
        if (err_ != hipSuccess) {                                       \
            die(std::string(#call) + ": " + hipGetErrorString(err_));   \
        }                                                               \
    } while (false)
void* device_alloc(std::size_t bytes)
{
    void* p = nullptr;
    DEVICE_CHECK(hipMalloc(&p, std::max<std::size_t>(bytes, 8)));
    return p;
}
void device_free(void* p) { DEVICE_CHECK(hipFree(p)); }
void to_device(void* dst, const void* src, std::size_t bytes)
{
    DEVICE_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}
void to_host(void* dst, const void* src, std::size_t bytes)
{
    DEVICE_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
}
const char* buffers = "hip";
#else
void* device_alloc(std::size_t bytes)
{
    void* p = std::malloc(std::max<std::size_t>(bytes, 8));
    if (!p) {
        die("out of memory");
    }
    return p;
}
void device_free(void* p) { std::free(p); }
void to_device(void* dst, const void* src, std::size_t bytes)
{
    std::memcpy(dst, src, bytes);
}
void to_host(void* dst, const void* src, std::size_t bytes)
{
    std::memcpy(dst, src, bytes);
}
const char* buffers = "host";
#endif


// An array of doubles in device memory.
class Buffer {
public:
    explicit Buffer(std::size_t count)
        : data_{static_cast<double*>(device_alloc(count * sizeof(double)))}
    {}
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { device_free(data_); }

    double* get() const { return data_; }

    void fill(std::size_t count, double value)
    {
        std::vector<double> host(count, value);
        to_device(data_, host.data(), count * sizeof(double));
    }

    double at(std::size_t i) const
    {
        double value;
        to_host(&value, data_ + i, sizeof(double));
        return value;
    }

private:
    double* data_;
};


struct Options {
    int max_allreduce_k = 14;  // 4^14 doubles = 2 GiB
    int max_alltoallv_k = 11;  // 4^11 doubles = 32 MiB per destination
    bool pingpong = false;
    long long max_pingpong_bytes = 1LL << 30;
    int ranks_per_node = 4;
    double min_time = 0.2;  // time budget per size, sets the repetitions
};


Options parse(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                die("missing value for " + arg);
            }
            return argv[++i];
        };
        if (arg == "--max-allreduce-k") {
            o.max_allreduce_k = std::stoi(value());
        } else if (arg == "--max-alltoallv-k") {
            o.max_alltoallv_k = std::stoi(value());
        } else if (arg == "--pingpong") {
            o.pingpong = true;
        } else if (arg == "--max-pingpong-bytes") {
            o.max_pingpong_bytes = std::stoll(value());
        } else if (arg == "--ranks-per-node") {
            o.ranks_per_node = std::stoi(value());
        } else if (arg == "--min-time") {
            o.min_time = std::stod(value());
        } else {
            die("unknown argument " + arg);
        }
    }
    return o;
}


std::size_t power4(int k) { return std::size_t{1} << (2 * k); }


std::string mpi_error(int code)
{
    char text[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(code, text, &length);
    return std::string(text, length);
}


struct Result {
    bool completed = true;
    std::string error;
    double time = 0.0;
    double min_time = 0.0;
    int repetitions = 0;
};


// One run of the operation; returns an MPI error code.
using Operation = std::function<int()>;
// Whether this rank's result of the last run is right.
using Check = std::function<bool()>;


Result measure(MPI_Comm comm, double min_time, const Operation& op,
               const Check& check)
{
    Result r;
    auto timed = [&](int& err) {
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        err = op();
        double t = MPI_Wtime() - start;
        double slowest = 0.0;
        MPI_Allreduce(&t, &slowest, 1, MPI_DOUBLE, MPI_MAX, comm);
        return slowest;
    };
    int err = MPI_SUCCESS;
    const double first = timed(err);  // warm-up, and its result is checked
    int ok = err == MPI_SUCCESS && check() ? 1 : 0;
    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
    if (!all_ok) {
        r.completed = false;
        r.error = err != MPI_SUCCESS ? mpi_error(err) : "wrong result";
        return r;
    }
    // the same on every rank: `first` is the slowest rank's time
    const int repetitions = std::clamp(
        static_cast<int>(std::ceil(min_time / std::max(first, 1e-9))), 3, 100);
    std::vector<double> times;
    for (int i = 0; i < repetitions; ++i) {
        times.push_back(timed(err));
        if (err != MPI_SUCCESS) {
            r.completed = false;
            r.error = mpi_error(err);
            return r;
        }
    }
    std::sort(times.begin(), times.end());
    const auto n = times.size();
    r.time = n % 2 ? times[n / 2] : (times[n / 2 - 1] + times[n / 2]) / 2;
    r.min_time = times.front();
    r.repetitions = repetitions;
    return r;
}


std::string escape(const std::string& s)
{
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        out += (c == '\n' || c == '\t') ? ' ' : c;
    }
    return out;
}


struct Case {
    std::string operation;
    std::size_t count;
    int ranks;
    Result result;
};


void print(const std::vector<Case>& cases)
{
    std::printf("[\n");
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        const auto& r = c.result;
        std::printf(
            "  {\"operation\": \"%s\", \"count\": %zu, \"bytes\": %zu, "
            "\"ranks\": %d, \"buffers\": \"%s\", \"mpi\": {\"%s\": "
            "{\"completed\": %s, \"time\": %.9g, \"min_time\": %.9g, "
            "\"repetitions\": %d",
            c.operation.c_str(), c.count, c.count * sizeof(double), c.ranks,
            buffers, c.operation.c_str(), r.completed ? "true" : "false",
            r.time, r.min_time, r.repetitions);
        if (!r.completed) {
            std::printf(", \"error\": \"%s\"", escape(r.error).c_str());
        }
        std::printf("}}}%s\n", i + 1 < cases.size() ? "," : "");
    }
    std::printf("]\n");
}


void progress(int rank, const Case& c)
{
    if (rank == 0) {
        std::fprintf(stderr, "%s %zu B: %s %.3g s\n", c.operation.c_str(),
                     c.count * sizeof(double),
                     c.result.completed ? "ok" : "FAILED", c.result.time);
    }
}


void allreduce(MPI_Comm comm, const Options& o, std::vector<Case>& cases)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    Buffer send(power4(o.max_allreduce_k));
    Buffer recv(power4(o.max_allreduce_k));
    const double expected = size * (size - 1) / 2.0;
    for (int k = 0; k <= o.max_allreduce_k; ++k) {
        const auto count = power4(k);
        send.fill(count, rank);
        Case c{"allreduce", count, size,
               measure(
                   comm, o.min_time,
                   [&] {
                       return MPI_Allreduce(send.get(), recv.get(),
                                            static_cast<int>(count), MPI_DOUBLE,
                                            MPI_SUM, comm);
                   },
                   [&] {
                       return recv.at(0) == expected &&
                              recv.at(count - 1) == expected;
                   })};
        progress(rank, c);
        cases.push_back(c);
    }
}


void alltoallv(MPI_Comm comm, const Options& o, std::vector<Case>& cases)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    Buffer send(power4(o.max_alltoallv_k) * size);
    Buffer recv(power4(o.max_alltoallv_k) * size);
    for (int k = 0; k <= o.max_alltoallv_k; ++k) {
        const auto count = power4(k);
        std::vector<int> counts(size, static_cast<int>(count));
        std::vector<int> displs(size);
        for (int i = 0; i < size; ++i) {
            displs[i] = static_cast<int>(i * count);
        }
        send.fill(count * size, rank);  // each block holds its source rank
        Case c{"alltoallv", count, size,
               measure(
                   comm, o.min_time,
                   [&] {
                       return MPI_Alltoallv(send.get(), counts.data(),
                                            displs.data(), MPI_DOUBLE,
                                            recv.get(), counts.data(),
                                            displs.data(), MPI_DOUBLE, comm);
                   },
                   [&] {
                       for (int i = 0; i < size; ++i) {
                           if (recv.at(displs[i]) != i ||
                               recv.at(displs[i] + count - 1) != i) {
                               return false;
                           }
                       }
                       return true;
                   })};
        progress(rank, c);
        cases.push_back(c);
    }
}


// The face neighbors on a 3D grid of the ranks, as the 7-point stencil's halo
// exchange has them.
std::vector<int> grid_neighbors(MPI_Comm comm)
{
    int size;
    MPI_Comm_size(comm, &size);
    int dims[3] = {0, 0, 0};
    int periods[3] = {0, 0, 0};
    MPI_Dims_create(size, 3, dims);
    MPI_Comm cart;
    MPI_Cart_create(comm, 3, dims, periods, 0, &cart);
    std::vector<int> neighbors;
    for (int d = 0; d < 3; ++d) {
        int lower, upper;
        MPI_Cart_shift(cart, d, 1, &lower, &upper);
        for (int n : {lower, upper}) {
            if (n != MPI_PROC_NULL) {
                neighbors.push_back(n);
            }
        }
    }
    MPI_Comm_free(&cart);
    return neighbors;
}


void neighbor_alltoallv(MPI_Comm comm, const Options& o,
                        std::vector<Case>& cases)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const auto neighbors = grid_neighbors(comm);
    const int n = static_cast<int>(neighbors.size());
    int dummy = 0;
    const int* list = n ? neighbors.data() : &dummy;
    MPI_Comm graph;
    MPI_Dist_graph_create_adjacent(comm, n, list, MPI_UNWEIGHTED, n, list,
                                   MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &graph);
    MPI_Comm_set_errhandler(graph, MPI_ERRORS_RETURN);
    const auto blocks = static_cast<std::size_t>(std::max(n, 1));
    Buffer send(power4(o.max_alltoallv_k) * blocks);
    Buffer recv(power4(o.max_alltoallv_k) * blocks);
    for (int k = 0; k <= o.max_alltoallv_k; ++k) {
        const auto count = power4(k);
        std::vector<int> counts(blocks, static_cast<int>(count));
        std::vector<int> displs(blocks);
        for (std::size_t i = 0; i < blocks; ++i) {
            displs[i] = static_cast<int>(i * count);
        }
        send.fill(count * blocks, rank);
        Case c{"neighbor_alltoallv", count, size,
               measure(
                   comm, o.min_time,
                   [&] {
                       return MPI_Neighbor_alltoallv(
                           send.get(), counts.data(), displs.data(), MPI_DOUBLE,
                           recv.get(), counts.data(), displs.data(), MPI_DOUBLE,
                           graph);
                   },
                   [&] {
                       for (int i = 0; i < n; ++i) {
                           if (recv.at(displs[i]) != neighbors[i] ||
                               recv.at(displs[i] + count - 1) != neighbors[i]) {
                               return false;
                           }
                       }
                       return true;
                   })};
        progress(rank, c);
        cases.push_back(c);
    }
    MPI_Comm_free(&graph);
}


// A ping-pong between ranks a and b; the others wait at the barriers.
void pingpong(MPI_Comm comm, const Options& o, const std::string& name, int a,
              int b, std::vector<Case>& cases)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const auto max_count =
        static_cast<std::size_t>(o.max_pingpong_bytes) / sizeof(double);
    Buffer buffer(max_count);
    for (std::size_t count = 1; count <= max_count; count *= 4) {
        buffer.fill(count, rank == a ? 1.0 : 0.0);
        const int n = static_cast<int>(count);
        auto result = measure(
            comm, o.min_time,
            [&] {
                int err = MPI_SUCCESS;
                if (rank == a) {
                    err = MPI_Send(buffer.get(), n, MPI_DOUBLE, b, 0, comm);
                    if (err == MPI_SUCCESS) {
                        err = MPI_Recv(buffer.get(), n, MPI_DOUBLE, b, 0, comm,
                                       MPI_STATUS_IGNORE);
                    }
                } else if (rank == b) {
                    err = MPI_Recv(buffer.get(), n, MPI_DOUBLE, a, 0, comm,
                                   MPI_STATUS_IGNORE);
                    if (err == MPI_SUCCESS) {
                        err = MPI_Send(buffer.get(), n, MPI_DOUBLE, a, 0, comm);
                    }
                }
                return err;
            },
            [&] {
                return rank != b ||
                       (buffer.at(0) == 1.0 && buffer.at(count - 1) == 1.0);
            });
        result.time /= 2;  // one way
        result.min_time /= 2;
        Case c{name, count, size, result};
        progress(rank, c);
        cases.push_back(c);
    }
}


}  // namespace


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    const auto options = parse(argc, argv);
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<Case> cases;
    allreduce(comm, options, cases);
    alltoallv(comm, options, cases);
    neighbor_alltoallv(comm, options, cases);
    if (options.pingpong) {
        if (size >= 2) {
            pingpong(comm, options, "pingpong_intra", 0, 1, cases);
        }
        if (size > options.ranks_per_node) {
            pingpong(comm, options, "pingpong_inter", 0, options.ranks_per_node,
                     cases);
        }
    }
    if (rank == 0) {
        print(cases);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
}
