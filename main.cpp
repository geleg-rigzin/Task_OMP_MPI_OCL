#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <omp.h>
#include <cmath>
#include <climits>
#define MAX_SIZE 1000000

struct CSRPortrait {
    int M = 0;
    int N = 0;
    std::vector <int> cumulative_sum;
    std::vector <int> column_numbers;
    int transpose(CSRPortrait &tCSR) const;
    int reserved_memory() const;
};

int CSRPortrait::transpose(CSRPortrait &tCSR) const {
    tCSR.M = this->N;
    tCSR.N = this->M;
    tCSR.cumulative_sum.clear();
    tCSR.column_numbers.clear();

    tCSR.cumulative_sum.resize(tCSR.M+2, 0);
    #pragma omp parallel
    {
        #pragma omp for
        for(auto i = 0; i < this->M; ++i) {
            for(auto k = this->cumulative_sum[i]; k < this->cumulative_sum[i+1]; ++k) {
                tCSR.cumulative_sum[this->column_numbers[k]+2]++;
            }
        }
        const int nt = omp_get_num_threads();
        const int tn = omp_get_thread_num();
        const int size_cum_sum = tCSR.M;
        int ibeg = tn*((size_cum_sum+1)/(nt+1)) + ( tn < (size_cum_sum+1)%(nt+1) ? tn : (size_cum_sum+1)%(nt+1) ) + 1;
        int iend = ibeg + ((size_cum_sum+1)/(nt+1)) + (tn < (size_cum_sum+1)%(nt+1));

        for(auto i = ibeg; i < iend-1; ++i) {
            tCSR.cumulative_sum[i+1] += tCSR.cumulative_sum[i];
            //portrait_M.cumulative_sum[i+1] += portrait_M.cumulative_sum[i];
        }
        #pragma omp barrier
        #pragma omp master
        for(auto i = 1; i < nt; ++i) {
            int ibeg = i*((size_cum_sum+1)/(nt+1)) + ( i < (size_cum_sum+1)%(nt+1) ? i : (size_cum_sum+1)%(nt+1) ) + 1;
            int iend = ibeg + ((size_cum_sum+1)/(nt+1)) + (i < (size_cum_sum+1)%(nt+1));
            tCSR.cumulative_sum[iend-1] += tCSR.cumulative_sum[ibeg-1];
        }
        #pragma omp barrier
        ibeg = iend;
        iend = ibeg + ((size_cum_sum+1)/(nt+1)) + ((tn+1) < (size_cum_sum+1)%(nt+1));

        if(tn != nt-1) {
            for(auto i = ibeg; i < iend-1; ++i) {
                tCSR.cumulative_sum[i] += tCSR.cumulative_sum[ibeg-1];
            }
        } else {
            for(auto i = ibeg; i < iend; ++i) {
                tCSR.cumulative_sum[i] += tCSR.cumulative_sum[i-1];
            }
        }
/*
        for(auto i = 1; i < tCSR.M+1; ++i) {
            tCSR.cumulative_sum[i+1] += tCSR.cumulative_sum[i];
        }
*/
        #pragma omp master
        {
            tCSR.column_numbers.resize(tCSR.cumulative_sum[tCSR.M+1], 0);
        }
        #pragma omp barrier
        #pragma omp for
        for(auto i = 0; i < this->M; ++i) {
            for(auto k = this->cumulative_sum[i]; k < this->cumulative_sum[i+1]; ++k) {
                tCSR.column_numbers[tCSR.cumulative_sum[this->column_numbers[k]+1]++] = i;
            }
        }
    }
    tCSR.cumulative_sum.pop_back();
    return 0;
}

int CSRPortrait::reserved_memory() const {
    return this->cumulative_sum.capacity()*sizeof(cumulative_sum[0]) + this->column_numbers.capacity()*sizeof(column_numbers[0]);
}

void print_CSRPortrait(const CSRPortrait &CSR) {
    std::cout << "Matrix " << CSR.M << " x " << CSR.N << std::endl;
    std::cout << "[";
    for(auto element : CSR.cumulative_sum) {
        std::cout << element << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << "[";
    for(auto element : CSR.column_numbers) {
        std::cout << element << " ";
    }
    std::cout << "]" << std::endl;
}

int generate_grid(int Nx, int Ny, int k1, int k2, CSRPortrait &EN) {

    EN.column_numbers.clear();
    EN.cumulative_sum.clear();
    const auto sum_k1_k2 = k1 + k2;
    const auto whole_part = (Nx-1)*(Ny-1)/sum_k1_k2, remainder = (Nx-1)*(Ny-1)%sum_k1_k2;
    const auto size_cum_sum = whole_part * (k1+2*k2) + (k1 + (remainder - k1) * 2)*(remainder >= k1) + (remainder < k1) * remainder;
    const auto size_col_num = whole_part * (k1*4+6*k2) + (k1*4 + (remainder - k1) * 6)*(remainder >= k1) + (remainder < k1) * 4 * remainder;

    EN.cumulative_sum.resize(size_cum_sum + 1, 0);
    EN.column_numbers.resize(size_col_num, 0);
    #pragma omp parallel
    {
        #pragma omp for
        for(auto i = 0; i < Ny-1; ++i) {
            for(auto j = 0; j < Nx-1; ++j) {
                const auto square = i*(Nx-1)+j;
                const auto whole_part = square/sum_k1_k2, remainder = square%sum_k1_k2;
                if(remainder < k1){
                    auto ind_cum_sum = whole_part * (k1 + 2*k2) + remainder;
                    auto ind_col_num = whole_part * (k1 * 4 + 6 * k2) + 4 * remainder;
                    EN.cumulative_sum[ind_cum_sum + 1] = 4;
                    EN.column_numbers[ind_col_num++] = i*Nx+j;
                    EN.column_numbers[ind_col_num++] = i*Nx+j+1;
                    EN.column_numbers[ind_col_num++] = (i+1)*Nx+j;
                    EN.column_numbers[ind_col_num] = (i+1)*Nx+j+1;
                }
                else {
                    auto ind_cum_sum = whole_part * (k1 + 2*k2) + (k1 + (remainder - k1) * 2);
                    auto ind_col_num = whole_part * (k1 * 4 + 6 * k2) + (k1 * 4 + (remainder - k1) * 6);
                    EN.cumulative_sum[ind_cum_sum + 1] = 3;
                    EN.column_numbers[ind_col_num++] = i*Nx+j;
                    EN.column_numbers[ind_col_num++] = i*Nx+j+1;
                    EN.column_numbers[ind_col_num++] = (i+1)*Nx+j;
                    EN.cumulative_sum[ind_cum_sum + 2] = 3;
                    EN.column_numbers[ind_col_num++] = i*Nx+j+1;
                    EN.column_numbers[ind_col_num++] = (i+1)*Nx+j;
                    EN.column_numbers[ind_col_num] = (i+1)*Nx+j+1;
                }
            }
        }
        const int nt = omp_get_num_threads();
        const int tn = omp_get_thread_num();
        int ibeg = tn*((size_cum_sum+1)/(nt+1)) + ( tn < (size_cum_sum+1)%(nt+1) ? tn : (size_cum_sum+1)%(nt+1) );
        int iend = ibeg + ((size_cum_sum+1)/(nt+1)) + (tn < (size_cum_sum+1)%(nt+1));

        for(auto i = ibeg; i < iend-1; ++i) {
            EN.cumulative_sum[i+1] += EN.cumulative_sum[i];
        }
        #pragma omp barrier
        #pragma omp master
        for(auto i = 1; i < nt; ++i) {
            int ibeg = i*((size_cum_sum+1)/(nt+1)) + ( i < (size_cum_sum+1)%(nt+1) ? i : (size_cum_sum+1)%(nt+1) );
            int iend = ibeg + ((size_cum_sum+1)/(nt+1)) + (i < (size_cum_sum+1)%(nt+1));
            EN.cumulative_sum[iend-1] += EN.cumulative_sum[ibeg-1];
        }
        #pragma omp barrier
        ibeg = iend;
        iend = ibeg + ((size_cum_sum+1)/(nt+1)) + ((tn+1) < (size_cum_sum+1)%(nt+1));

        if(tn != nt-1) {
            for(auto i = ibeg; i < iend-1; ++i) {
                EN.cumulative_sum[i] += EN.cumulative_sum[ibeg-1];
            }
        } else {
            for(auto i = ibeg; i < iend; ++i) {
                EN.cumulative_sum[i] += EN.cumulative_sum[i-1];
            }
        }
    }
    EN.M = size_cum_sum;
    EN.N = Nx*Ny;
    return EN.reserved_memory();
}

int construct_matrix_E_f_E_old(const CSRPortrait &EN, CSRPortrait &EfE) {
    CSRPortrait NE;
    EN.transpose(NE);
    EfE.M = EfE.N = EN.M;
    EfE.cumulative_sum.clear();
    EfE.column_numbers.clear();
    EfE.cumulative_sum.resize(EfE.M+1, 0);
    std::set<int> set_of_edges;
    std::vector<int> vector_to_intersect(EN.N);
    std::vector<int>::iterator end_iter_of_interruption;
    int max_size_of_set = 0, current_size_of_set = 0;
    for(auto i = 0; i < EN.M; ++i) {
        set_of_edges.clear();
        auto begin_i_str = EN.cumulative_sum[i];
        auto end_i_str = EN.cumulative_sum[i+1];
        for(auto k = begin_i_str; k < end_i_str; ++k) {
            auto j = EN.column_numbers[k];
            for(auto c = NE.cumulative_sum[j]; c < NE.cumulative_sum[j+1]; ++c) {
                set_of_edges.insert(NE.column_numbers[c]);
            }
        }
        current_size_of_set = set_of_edges.size();
        max_size_of_set = (current_size_of_set>max_size_of_set) ? current_size_of_set : max_size_of_set;

        auto begin_iter_i_str = EN.column_numbers.begin() + begin_i_str;
        auto end_iter_i_str = EN.column_numbers.begin() + end_i_str;
        for(auto edge : set_of_edges) {
            auto begin_iter_cur_str = EN.column_numbers.begin() + EN.cumulative_sum[edge];
            auto end_iter_cur_str = EN.column_numbers.begin() + EN.cumulative_sum[edge+1];
            end_iter_of_interruption = std::set_intersection(begin_iter_i_str, end_iter_i_str, begin_iter_cur_str, end_iter_cur_str, vector_to_intersect.begin());
            if(end_iter_of_interruption - vector_to_intersect.begin() > 1) {
                ++EfE.cumulative_sum[i+1];
                EfE.column_numbers.push_back(edge);
            }
        }
    }
    for(auto i = 0; i < EfE.M; ++i) {
        EfE.cumulative_sum[i+1] += EfE.cumulative_sum[i];
    }

    return max_size_of_set*sizeof(set_of_edges) + NE.reserved_memory() + EfE.reserved_memory() + vector_to_intersect.capacity()*sizeof(vector_to_intersect[0]);
}
int construct_matrix_E_f_E(const CSRPortrait &EN, CSRPortrait &EfE) {
    CSRPortrait NE;
    EN.transpose(NE);
    EfE.M = EfE.N = EN.M;
    EfE.cumulative_sum.clear();
    EfE.column_numbers.clear();
    EfE.cumulative_sum.resize(EfE.M+1, 0);
    std::vector<int> vector_to_intersect;
    for(auto i = 0; i < EN.M; ++i) {
        vector_to_intersect.clear();
        const auto begin_i_str = EN.cumulative_sum[i];
        const auto end_i_str = EN.cumulative_sum[i+1];
        for(auto k = begin_i_str; k < end_i_str; ++k) {
            auto j = EN.column_numbers[k];
            for(auto c = NE.cumulative_sum[j]; c < NE.cumulative_sum[j+1]; ++c) {
                vector_to_intersect.push_back(NE.column_numbers[c]);
            }
        }
        std::sort(vector_to_intersect.begin(), vector_to_intersect.end());
        for(auto x : vector_to_intersect) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        auto prev = -1;
        for(auto j = 1; j < (int)vector_to_intersect.size(); ++j) {
            if((vector_to_intersect[j] != prev) && (vector_to_intersect[j] == vector_to_intersect[j-1])) {
                ++EfE.cumulative_sum[i+1];
                std::cout << prev << " " << vector_to_intersect[j] << std::endl;
                prev = vector_to_intersect[j];
                EfE.column_numbers.push_back(prev);
                ++j;
            }
        }
    }
    for(auto i = 0; i < EfE.M; ++i) {
        EfE.cumulative_sum[i+1] += EfE.cumulative_sum[i];
    }

    return NE.reserved_memory() + EfE.reserved_memory() + vector_to_intersect.capacity()*sizeof(vector_to_intersect[0]);
}

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)>"
              << "Options:\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "\t-d,--debug debugging\tOutput of the adjacency matrix in csr format\n"
              << "\t-Nx <int>, \tthe number of nodes on the 'x' axis, there must be more than one, required parameter\n"
              << "\t-Ny <int>, \tthe number of nodes on the 'y' axis, there must be more than one, required parameter\n"
              << "\t-k1 <int>, \tnumber of square cells, in total, k1+k2 must be greater than zero, required parameter\n"
              << "\t-k2 <int>, \tthe number of triangular cells, in total, k1+k2 must be greater than zero, required parameter\n"
              << std::endl;
}

int initialization_A(const CSRPortrait &portrait, std::vector <double> &A_value) {
    A_value.clear();
    A_value.resize(portrait.column_numbers.size());
    #pragma omp parallel for
    //#pragma omp parallel for schedule(dynamic, 100)
    for(auto i = 0; i < portrait.M; ++i) {
        auto str_sum = 0.;
        auto diag_offset = -1;
        for(auto k = portrait.cumulative_sum[i]; k < portrait.cumulative_sum[i+1]; ++k) {
            auto j = portrait.column_numbers[k];
            if(i != j) {
                //A_value[k] = std::sin(i + j);
                A_value[k] = std::cos(i * j + i + j);
                str_sum += std::abs(A_value[k]);
            }
            else {
                diag_offset = k;
            }
        }
        if(diag_offset >= 0) {
            A_value[diag_offset] = 1.234 * str_sum;
        }
    }

    return A_value.empty()? 0 : A_value.capacity()*sizeof(A_value[0]);
}

int fill_in_b(std::vector <double> &b, int size_of_b) {
    b.clear();
    b.resize(size_of_b);
    #pragma omp parallel for
    for(auto i = 0; i < size_of_b; ++i) {
        b[i] = std::sin(i);
    }
    return b.empty()? 0 : b.capacity()*sizeof(b[0]);
}

int SpMV(const CSRPortrait &portrait,
         const std::vector <double> &A_value,
         const std::vector <double> &vector1,
         std::vector <double> &result_vector) {
    #pragma omp parallel for
    //#pragma omp parallel for schedule(dynamic, 100)
    for(auto i = 0; i < portrait.M; ++i) {
        auto sum = 0.;
        for(auto k = portrait.cumulative_sum[i]; k < portrait.cumulative_sum[i+1]; ++k) {
            sum += A_value[k] * vector1[portrait.column_numbers[k]];
        }
        result_vector[i] = sum;
    }
    return 0;
}

int axpy(const std::vector <double> &vector_1,
         const std::vector <double> &vector_2,
         const double scalar,
         std::vector <double> &result_vector) {
    //проверку
    if(vector_1.size() != vector_2.size() || vector_1.size() != result_vector.size()) {
        std::cout << "Vector size in function axpy not eq" << std::endl;
        exit(10);
    }
    #pragma omp parallel for
    for(auto i = 0; i < (int)vector_1.size(); ++i) {
        result_vector[i] = vector_1[i] + scalar * vector_2[i];
    }

    return 0;
}

double dot(const std::vector <double> &vector_1,
           const std::vector <double> &vector_2) {
    auto result = 0.;
    #pragma omp parallel for reduction(+ : result)
    for(auto i = 0; i < (int)vector_1.size(); ++i) {
        result += vector_1[i] * vector_2[i];
    }
    return result;
}

int construct_matrix_M(const CSRPortrait &portrait_A,
                       const std::vector <double> &A_value,
                       CSRPortrait &portrait_M,
                       std::vector <double> &M_value) {

    portrait_M.cumulative_sum.clear();
    portrait_M.column_numbers.clear();
    portrait_M.M = portrait_A.M;
    portrait_M.N = portrait_A.N;
    portrait_M.cumulative_sum.resize(portrait_M.M + 1, 0);
    portrait_M.column_numbers.resize(portrait_M.N, 0);
    M_value.resize(portrait_M.N, 0);
    #pragma omp parallel
    {
        #pragma omp for
        for(auto i = 0; i < portrait_A.M; ++i) {
            for(auto k = portrait_A.cumulative_sum[i]; k < portrait_A.cumulative_sum[i+1]; ++k) {
                auto j = portrait_A.column_numbers[k];
                if(i == j) {
                    M_value[i] = A_value[k];
                    portrait_M.column_numbers[i] = i;
                    ++portrait_M.cumulative_sum[i+1];
                }
            }
        }
        const int nt = omp_get_num_threads();
        const int tn = omp_get_thread_num();
        const int size_cum_sum = portrait_M.M;
        int ibeg = tn*((size_cum_sum+1)/(nt+1)) + ( tn < (size_cum_sum+1)%(nt+1) ? tn : (size_cum_sum+1)%(nt+1) );
        int iend = ibeg + ((size_cum_sum+1)/(nt+1)) + (tn < (size_cum_sum+1)%(nt+1));

        for(auto i = ibeg; i < iend-1; ++i) {
            portrait_M.cumulative_sum[i+1] += portrait_M.cumulative_sum[i];
        }
        #pragma omp barrier
        #pragma omp master
        for(auto i = 1; i < nt; ++i) {
            int ibeg = i*((size_cum_sum+1)/(nt+1)) + ( i < (size_cum_sum+1)%(nt+1) ? i : (size_cum_sum+1)%(nt+1) );
            int iend = ibeg + ((size_cum_sum+1)/(nt+1)) + (i < (size_cum_sum+1)%(nt+1));
            portrait_M.cumulative_sum[iend-1] += portrait_M.cumulative_sum[ibeg-1];
        }
        #pragma omp barrier
        ibeg = iend;
        iend = ibeg + ((size_cum_sum+1)/(nt+1)) + ((tn+1) < (size_cum_sum+1)%(nt+1));

        if(tn != nt-1) {
            for(auto i = ibeg; i < iend-1; ++i) {
                portrait_M.cumulative_sum[i] += portrait_M.cumulative_sum[ibeg-1];
            }
        } else {
            for(auto i = ibeg; i < iend; ++i) {
                portrait_M.cumulative_sum[i] += portrait_M.cumulative_sum[i-1];
            }
        }
    }
    return 0;
}

int solve(const CSRPortrait &portrait_A,
          const std::vector <double> &A_value,
          const std::vector <double> &b,
          const double eps,
          const int maxit,
          std::vector <double> &x) {
    CSRPortrait portrait_M;
    std::vector <double> M_value;
    construct_matrix_M(portrait_A, A_value, portrait_M, M_value);
    double rho_k = 0, rho_k_1 = 0, eps_q = eps*eps;
    std::vector <double> r = b; //    𝒓0 = 𝒃

    int k = 0;                  //    𝑘 = 0
    x.clear();
    x.resize(portrait_A.M, 0);  //    𝒙0 = 0
    std::vector <double> z(portrait_A.M);
    std::vector <double> p(portrait_A.M);
    std::vector <double> q(portrait_A.M);
    do {
        ++k;
        SpMV(portrait_M, M_value, r, z); //𝒛𝑘 = 𝑴−1𝒓𝑘−1 // SpMV

        rho_k = dot(r, z);               //𝜌𝑘 = (𝒓𝑘−1, 𝒛𝑘) // dot
        if (k == 1) {                    //if 𝑘 = 1 then
            p = z;                       //𝒑𝑘 = 𝒛𝑘
        } else {
            auto beta = rho_k / rho_k_1; //𝛽𝑘 = 𝜌𝑘/𝜌𝑘−1
            axpy(z, p, beta, p);         //𝒑𝑘 = 𝒛𝑘 + 𝛽𝑘𝒑𝑘−1 // axpy
        }
        SpMV(portrait_A, A_value, p, q); //𝒒𝑘 = 𝑨𝒑𝑘 // SpMV
        auto alpha = rho_k / dot(p, q);  //𝛼𝑘 = 𝜌𝑘/(𝒑𝑘, 𝒒𝑘) // dot
        axpy(x, p, alpha, x);            //𝒙𝑘 = 𝒙𝑘−1 + 𝛼𝑘𝒑𝑘 // axpy
        axpy(r, q, alpha*(-1), r);       //𝒓𝑘 = 𝒓𝑘−1 − 𝛼𝑘𝒒𝑘 // axpy
        rho_k_1 = rho_k;
        if(k % 1 == 0) {
            SpMV(portrait_A, A_value, x, z);
            axpy(z, b, -1, z);
            std::cout << "iter N" << k << " " << dot(z,z) << " " << rho_k << std::endl;
        }
    } while (rho_k > eps_q && k < maxit);//𝜌𝑘 > 𝜀^2 and k < maxit

    SpMV(portrait_A, A_value, x, z);
    std::cout << "x :\n";
    for(auto z : x) {
        std::cout << z << " ";
    }
    std::cout << "z :\n";
    for(auto x : z) {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    std::cout << "b :\n";
    for(auto x : b) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    axpy(z, b, -1, z);
    std::cout << "iter last" << k << " " << dot(z,z) << " " << rho_k << std::endl;

    return 0;
}

int args_parsing(int argc, char *argv[], std::vector <int> &args_values) {
    args_values.clear();

    int Nx = 0, Ny = 0, k1 = 0, k2 = 0;
    bool debug_mode = false, args_exist[] = {false,false,false,false};
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "-Nx") {
            if (i + 1 < argc) {
                Nx = strtol(argv[++i], nullptr, 0);
                if(Nx <= 1 || Nx >= MAX_SIZE) {
                    std::cerr << "-Nx value most be greater then 1 and less than " << MAX_SIZE << std::endl;
                    exit(1);
                }
                args_exist[0] = true;
            } else {
                std::cerr << "-Nx option requires one argument." << std::endl;
                exit(1);
            }
        } else if(arg == "-Ny") {
            if (i + 1 < argc) {
                Ny = strtol(argv[++i], nullptr, 0);
                if(Ny <= 1 || Ny >= MAX_SIZE) {
                    std::cerr << "-Ny value most be greater then 1 and less than " << MAX_SIZE << std::endl;
                    exit(2);
                }
                args_exist[1] = true;
            } else {
                std::cerr << "-Ny option requires one argument." << std::endl;
                exit(2);
            }
        } else if(arg == "-k1") {
            if (i + 1 < argc) {
                k1 = strtol(argv[++i], nullptr, 0);
                if(k1 < 0) {
                    std::cerr << "-k1 value must be at least 0 " <<std::endl;
                    exit(3);
                }
                args_exist[2] = true;
            } else {
                std::cerr << "-k1 value requires one argument." << std::endl;
                exit(3);
            }
        } else if(arg == "-k2") {
            if (i + 1 < argc) {
                k2 = strtol(argv[++i], nullptr, 0);
                if(k2 < 0) {
                    std::cerr << "-k2 value must be at least 0 " <<std::endl;
                    exit(4);
                }
                args_exist[3] = true;
            } else {
                std::cerr << "-k2 option requires one argument." << std::endl;
                exit(4);
            }
        } else if(arg == "-d" || arg == "--debug") {
                debug_mode = true;
        } else if(arg == "-h"||arg == "--help") {
            show_usage(argv[0]);
            exit(0);
        } else if(arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                auto num_threads = strtol(argv[++i], nullptr, 0);
                omp_set_num_threads(num_threads);
            }
        }
    }
    if(debug_mode && Nx*Ny > 100) {
        std::cout << "The size of the matrix is too large to display on the screen, debugging mode is disabled" << std::endl;
        debug_mode = false;
    }
    if(args_exist[0]*args_exist[1]*args_exist[2]*args_exist[3]*Nx*Ny == 0 || k1+k2 == 0) {
        std::cout << "Sorry, several required arguments are not specified or k1+k2=0" << std::endl;
        std::cout << "Try to read help." <<std::endl;
        show_usage(argv[0]);
        exit(5);
    }
    args_values.resize(5);
    args_values[0] = Nx;
    args_values[1] = Ny;
    args_values[2] = k1;
    args_values[3] = k2;
    args_values[4] = debug_mode;
    return 0;
}

int main(int argc, char *argv[]) {

    std::vector <int> args_values;
    args_parsing(argc, argv, args_values);
    int Nx = args_values[0], Ny = args_values[1], k1 = args_values[2], k2 = args_values[3];
    bool debug_mode = args_values[4];

    CSRPortrait newcsr, tcsr;
    int occupied_memory = 0;
    double start = omp_get_wtime();
    occupied_memory = generate_grid(Nx, Ny, k1, k2, newcsr);
    double stop = omp_get_wtime();
    std::cout << "Grid generation time:            " << stop - start << std::endl;
    std::cout << "Used memory for grid generation: " << occupied_memory << std::endl;

    start = omp_get_wtime();
    occupied_memory = construct_matrix_E_f_E(newcsr, tcsr);
    stop = omp_get_wtime();
    std::cout << "EfE matrix generation time:      " << stop - start << std::endl;
    std::cout << "Used memory for EfE generation:  " << occupied_memory << std::endl;

    std::vector <double> A_value;
    initialization_A(tcsr, A_value);
    std::vector <double> b;
    fill_in_b(b, tcsr.M);

    std::vector <double> x;
    auto eps = 0.0001;
    auto maxit = 100;
    solve(tcsr, A_value, b, eps, maxit, x);

    if(debug_mode) {
        print_CSRPortrait(newcsr);
        print_CSRPortrait(tcsr);
        for(auto x : A_value) {
            std::cout << x << ' ';
        }
        std::cout << std::endl;
    }

    return 0;
}
