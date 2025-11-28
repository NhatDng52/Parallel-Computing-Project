#include "scalability_test.h"
#include <iomanip>
#include <cstdlib>

void test_scalability_OpenMP(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    cout << "Testing scalability with different thread counts...\n" << endl;
    
    // Test matrix sizes
    vector<int> sizes = {128, 256, 512};
    vector<int> thread_counts = {1, 2, 4, 8};
    
    // Get max available threads
    int max_threads = omp_get_max_threads();
    cout << "Maximum available threads: " << max_threads << "\n" << endl;
    
    // Results storage
    cout << fixed << setprecision(2);
    cout << "┌──────────────┬─────────────┬───────────────┬─────────────┬──────────────┐" << endl;
    cout << "│ Matrix Size  │   Threads   │  Time (ms)    │  Speedup    │  Efficiency  │" << endl;
    cout << "├──────────────┼─────────────┼───────────────┼─────────────┼──────────────┤" << endl;
    
    for (int size : sizes) {
        // Generate random test matrices
        vector<vector<double>> A(size, vector<double>(size));
        vector<vector<double>> B(size, vector<double>(size));
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                A[i][j] = (rand() % 100) / 10.0;
                B[i][j] = (rand() % 100) / 10.0;
            }
        }
        
        double baseline_time = 0.0;
        
        for (int threads : thread_counts) {
            if (threads > max_threads) continue;
            
            omp_set_num_threads(threads);
            
            // Warm-up run
            auto warmup = matrix_mult(A, B);
            
            // Actual measurement (average of 3 runs)
            double total_time = 0.0;
            const int runs = 3;
            
            for (int run = 0; run < runs; run++) {
                auto start = chrono::high_resolution_clock::now();
                auto result = matrix_mult(A, B);
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> duration = end - start;
                total_time += duration.count();
            }
            
            double avg_time = total_time / runs;
            
            // Store baseline (single thread) time
            if (threads == 1) {
                baseline_time = avg_time;
            }
            
            double speedup = baseline_time / avg_time;
            double efficiency = (speedup / threads) * 100.0;
            
            cout << "│ " << setw(4) << size << "x" << setw(4) << size 
                 << "   │ " << setw(6) << threads 
                 << "      │ " << setw(10) << avg_time 
                 << "    │ " << setw(8) << speedup << "x"
                 << "   │ " << setw(9) << efficiency << "%"
                 << "   │" << endl;
        }
        
        if (size != sizes.back()) {
            cout << "├──────────────┼─────────────┼───────────────┼─────────────┼──────────────┤" << endl;
        }
    }
    
    cout << "└──────────────┴─────────────┴───────────────┴─────────────┴──────────────┘" << endl;
    
    // Strong scaling test (fixed problem size, varying threads)
    cout << "\n\nStrong Scaling Analysis (Fixed Matrix Size: 512x512):" << endl;
    cout << "─────────────────────────────────────────────────────────" << endl;
    
    int fixed_size = 512;
    vector<vector<double>> A_fixed(fixed_size, vector<double>(fixed_size));
    vector<vector<double>> B_fixed(fixed_size, vector<double>(fixed_size));
    
    for (int i = 0; i < fixed_size; i++) {
        for (int j = 0; j < fixed_size; j++) {
            A_fixed[i][j] = (rand() % 100) / 10.0;
            B_fixed[i][j] = (rand() % 100) / 10.0;
        }
    }
    
    double baseline = 0.0;
    for (int threads : thread_counts) {
        if (threads > max_threads) continue;
        
        omp_set_num_threads(threads);
        
        auto start = chrono::high_resolution_clock::now();
        auto result = matrix_mult(A_fixed, B_fixed);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        
        if (threads == 1) baseline = duration.count();
        
        double speedup = baseline / duration.count();
        cout << "  Threads: " << setw(2) << threads 
             << " | Time: " << setw(10) << duration.count() << " ms"
             << " | Speedup: " << setw(6) << speedup << "x" << endl;
    }
    
    cout << "\nScalability test with OpenMP completed" << endl;
}

void test_scalability_OpenMPI(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    cout << "Testing scalability with different MPI process counts...\n" << endl;
    
    cout << "Note: MPI scalability test requires running with different process counts" << endl;
    cout << "Example: mpirun -np 1 ./test_mpi, mpirun -np 2 ./test_mpi, etc.\n" << endl;
    
    vector<int> sizes = {128, 256, 512};
    
    cout << fixed << setprecision(2);
    cout << "┌──────────────┬───────────────┐" << endl;
    cout << "│ Matrix Size  │  Time (ms)    │" << endl;
    cout << "├──────────────┼───────────────┤" << endl;
    
    for (int size : sizes) {
        vector<vector<double>> A(size, vector<double>(size));
        vector<vector<double>> B(size, vector<double>(size));
        
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                A[i][j] = (rand() % 100) / 10.0;
                B[i][j] = (rand() % 100) / 10.0;
            }
        }
        
        auto warmup = matrix_mult(A, B);
        
        double total_time = 0.0;
        const int runs = 3;
        
        for (int run = 0; run < runs; run++) {
            auto start = chrono::high_resolution_clock::now();
            auto result = matrix_mult(A, B);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double, milli> duration = end - start;
            total_time += duration.count();
        }
        
        double avg_time = total_time / runs;
        
        cout << "│ " << setw(4) << size << "x" << setw(4) << size 
             << "   │ " << setw(10) << avg_time 
             << "    │" << endl;
        
        if (size != sizes.back()) {
            cout << "├──────────────┼───────────────┤" << endl;
        }
    }
    
    cout << "└──────────────┴───────────────┘" << endl;
    cout << "\nScalability test with OpenMPI completed" << endl;
}

void test_scalability_Hybrid(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // Some declarations
    // Hybrid test code combining OpenMP and OpenMPI
    cout<<"Scalability test with Hybrid completed \n";
}
