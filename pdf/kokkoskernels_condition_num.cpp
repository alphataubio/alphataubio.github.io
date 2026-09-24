#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_spiluk.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_copy.hpp>
#include <iostream>

using Scalar = double;
using Device = Kokkos::DefaultExecutionSpace;
using MemorySpace = Device::memory_space;
using ViewVector = Kokkos::View<Scalar*>;
using RowMapView = Kokkos::View<int*>;
using EntriesView = Kokkos::View<int*>;
using ValuesView = Kokkos::View<Scalar*>;
using Matrix = KokkosSparse::CrsMatrix<Scalar, int, Device, void, int>;

constexpr int max_iters = 100;
constexpr double tol = 1e-6;

// Power Iteration for Largest Singular Value
Scalar largestSingularValue(const Matrix& A) {
    int N = A.numRows();
    ViewVector x("x", N), Ax("Ax", N);
    
    Kokkos::parallel_for("Init", N, KOKKOS_LAMBDA(int i) { x(i) = 1.0; });
    
    Scalar sigma_old = 0.0, sigma = 0.0;
    for (int iter = 0; iter < max_iters; ++iter) {
        KokkosSparse::spmv("N", 1.0, A, x, 0.0, Ax); // Ax = A * x
        sigma = KokkosBlas::nrm2(Ax); // ||Ax||
        
        if (fabs(sigma - sigma_old) < tol) break;
        sigma_old = sigma;
        KokkosBlas::scal(x, 1.0 / sigma, Ax); // x = Ax / sigma
    }
    return sigma;
}

// Inverse Power Iteration for Smallest Singular Value (Approximate)
Scalar smallestSingularValue(const Matrix& A) {
    int N = A.numRows();
    ViewVector x("x", N), Ax("Ax", N);

    Kokkos::parallel_for("Init", N, KOKKOS_LAMBDA(int i) { x(i) = 1.0; });

    Scalar sigma_old = 0.0, sigma = 0.0;
    for (int iter = 0; iter < max_iters; ++iter) {
        KokkosSparse::spmv("N", 1.0, A, x, 0.0, Ax); // Ax = A * x
        KokkosBlas::scal(x, 1.0 / KokkosBlas::nrm2(Ax), Ax); // Normalize x
        
        sigma = KokkosBlas::nrm2(Ax);
        if (fabs(sigma - sigma_old) < tol) break;
        sigma_old = sigma;
    }
    return 1.0 / sigma; // Approximate inverse of smallest singular value
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int N = 3;
        RowMapView row_map("row_map", N + 1);
        EntriesView entries("entries", 5);
        ValuesView values("values", 5);

        // Example Sparse Matrix
        Kokkos::deep_copy(row_map, std::vector<int>{0, 2, 4, 5}.data());
        Kokkos::deep_copy(entries, std::vector<int>{0, 1, 0, 1, 2}.data());
        Kokkos::deep_copy(values, std::vector<Scalar>{4, 2, 3, 1, 5}.data());

        Matrix A("A", N, N, 5, values, row_map, entries);

        // Compute condition number
        Scalar sigma_max = largestSingularValue(A);
        Scalar sigma_min = smallestSingularValue(A);

        if (sigma_min > 0) {
            std::cout << "Condition Number: " << sigma_max / sigma_min << std::endl;
        } else {
            std::cout << "Matrix is singular (or ill-conditioned)." << std::endl;
        }
    }
    Kokkos::finalize();
    return 0;
}
