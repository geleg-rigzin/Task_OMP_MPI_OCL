#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <omp.h>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstring>
#include <CL/cl.h>
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

    for(auto i = 0; i < this->M; ++i) {
        for(auto k = this->cumulative_sum[i]; k < this->cumulative_sum[i+1]; ++k) {
            tCSR.cumulative_sum[this->column_numbers[k]+2]++;
        }
    }

    for(auto i = 1; i < tCSR.M+1; ++i) {
        tCSR.cumulative_sum[i+1] += tCSR.cumulative_sum[i];
    }
    tCSR.column_numbers.resize(tCSR.cumulative_sum[tCSR.M+1], 0);

    for(auto i = 0; i < this->M; ++i) {
        for(auto k = this->cumulative_sum[i]; k < this->cumulative_sum[i+1]; ++k) {
            tCSR.column_numbers[tCSR.cumulative_sum[this->column_numbers[k]+1]++] = i;
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
        #pragma omp master
        if(nt >= (int)EN.cumulative_sum.size()) {
            std::cout << "To many threads, restart with less parametr -t" << std::endl;
            exit(11);
        }
        #pragma omp barrier
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
        auto prev = -1;
        for(auto j = 1; j < (int)vector_to_intersect.size(); ++j) {
            if((vector_to_intersect[j] != prev) && (vector_to_intersect[j] == vector_to_intersect[j-1])) {
                ++EfE.cumulative_sum[i+1];
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
              << "\t-t, --threads <int>, \tthe number of thread must be greater than zero\n"
              << "\t-e, --epsilon <double>, \t must be greater than zero\n"
              << std::endl;
}

int initialization_A(const CSRPortrait &portrait, std::vector <double> &A_value) {
    A_value.clear();
    A_value.resize(portrait.column_numbers.size());
    #pragma omp parallel for
    //#pragma omp parallel for schedule(dynamic, 100)
    for(auto i = 0; i < portrait.M; ++i) {
        double str_sum = 0.;
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
        double sum = 0.;
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
    double result = 0.;
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
                    if(A_value[k] != 0)
                        M_value[i] = 1/A_value[k];
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
          std::vector <double> &x,
          std::vector <double> &times_of_kernels,
          bool debug) {
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
        double start = omp_get_wtime();
        SpMV(portrait_M, M_value, r, z); //𝒛𝑘 = 𝑴−1𝒓𝑘−1 // SpMV
        double stop = omp_get_wtime();
        times_of_kernels[0] += stop - start;
        start = omp_get_wtime();
        rho_k = dot(r, z);               //𝜌𝑘 = (𝒓𝑘−1, 𝒛𝑘) // dot
        stop = omp_get_wtime();
        times_of_kernels[1] += stop - start;

        if (k == 1) {                    //if 𝑘 = 1 then
            p = z;                       //𝒑𝑘 = 𝒛𝑘
        } else {
            double beta = rho_k / rho_k_1; //𝛽𝑘 = 𝜌𝑘/𝜌𝑘−1
            start = omp_get_wtime();
            axpy(z, p, beta, p);         //𝒑𝑘 = 𝒛𝑘 + 𝛽𝑘𝒑𝑘−1 // axpy
            stop = omp_get_wtime();
            times_of_kernels[2] += stop - start;
        }
        start = omp_get_wtime();
        SpMV(portrait_A, A_value, p, q); //𝒒𝑘 = 𝑨𝒑𝑘 // SpMV
        stop = omp_get_wtime();
        times_of_kernels[0] += stop - start;

        start = omp_get_wtime();
        double alpha = rho_k / dot(p, q);  //𝛼𝑘 = 𝜌𝑘/(𝒑𝑘, 𝒒𝑘) // dot
        stop = omp_get_wtime();
        times_of_kernels[1] += stop - start;
        start = omp_get_wtime();
        axpy(x, p, alpha, x);            //𝒙𝑘 = 𝒙𝑘−1 + 𝛼𝑘𝒑𝑘 // axpy
        axpy(r, q, alpha*(-1), r);       //𝒓𝑘 = 𝒓𝑘−1 − 𝛼𝑘𝒒𝑘 // axpy
        stop = omp_get_wtime();
        times_of_kernels[2] += stop - start;
        rho_k_1 = rho_k;
        if(k%10 == 0) {
            std::cout << "Iteration: " << k << ", rho: " << rho_k << std::endl; 
        }

    } while (rho_k > eps_q && k < maxit);//𝜌𝑘 > 𝜀^2 and k < maxit

    times_of_kernels[0] /= 2*k;
    times_of_kernels[1] /= 2*k;
    times_of_kernels[2] /= 3*k;
    return k;
}

void ocl_init(const cl_int devID, // Номер нужного девайса
              const char *platformName, //Нужная платформа // "Intel(R) OpenCL" // "NVIDIA CUDA"
              // OpenCL переменные
              cl_context &clContext, // OpenCL контекст
              cl_command_queue &clQueue, // OpenCL очередь команд
              cl_program &clProgram, // OpenCL программа 
              cl_int &clErr ) {// код возврата из OpenCL функций 
    
    //~ const cl_int devID = 0; // Номер нужного девайса
    //~ const char *platformName = "AMD Accelerated Parallel Processing"; //Нужная платформа // "Intel(R) OpenCL" // "NVIDIA CUDA"
    //~ // OpenCL переменные
    //~ cl_context clContext; // OpenCL контекст
    //~ cl_command_queue clQueue; // OpenCL очередь команд
    //~ cl_program clProgram; // OpenCL программа
    //~ cl_int clErr; // код возврата из OpenCL функций

    {// Инициализация OpenCL
        printf("OpenCL initialization\n");
        //PLATFORM
        // узнаем число платформ в системе
        cl_uint platformCount=0;
        clErr = clGetPlatformIDs( 0, 0, &platformCount);
        if(clErr != CL_SUCCESS){ printf("clGetPlatformIDs error %d\n", clErr); exit(1); }
        if(platformCount <= 0){ printf("No platforms found\n"); exit(1); }
        printf("clGetPlatformIDs: %d platforms\n", platformCount);

        // запрашиваем список платформ
        cl_platform_id *platformList = new cl_platform_id[platformCount];
        clErr = clGetPlatformIDs(platformCount, platformList, 0);
        if(clErr != CL_SUCCESS){ printf("clGetPlatformIDs error %d\n", clErr); exit(1); }
        // ищем нужную платформу
        #define STR_SIZE 1024 // размерчик буфера для названия платформ
        char nameBuf[STR_SIZE]; // буфер для названия платформы
        cl_int platform_id=0; // должно  быть меньше нуля
        //~ cl_int platform_id=1; // должно  быть меньше нуля
        for(cl_uint i=0; i<platformCount; i++){
            clErr = clGetPlatformInfo(platformList[i], CL_PLATFORM_NAME, STR_SIZE, nameBuf, 0);
            if(clErr != CL_SUCCESS){ printf("clGetPlatformInfo error %d\n", clErr); exit(1); }
            printf(" Platform %d: %s\n", i, nameBuf);
            if(!strcmp(platformName, nameBuf)) platform_id=i; // found
        }
        if(platform_id<0){ printf("Can't find platform\n"); exit(1); }
        printf("Platform %d selected\n",platform_id);

        // DEVICE
        // узнаем число девайсов у выбранной платформы
        int deviceCount = 0;
        clErr = clGetDeviceIDs(platformList[platform_id], CL_DEVICE_TYPE_ALL,
                               0, NULL,(cl_uint *) &deviceCount);
        if(clErr != CL_SUCCESS){
            switch(clErr) {
                case CL_INVALID_PLATFORM:
                    printf("CL_INVALID_PLATFORM\n");
                    break;
                case CL_INVALID_DEVICE_TYPE:
                    printf("CL_INVALID_DEVICE_TYPE\n");
                    break;
                case CL_INVALID_VALUE:
                    printf("CL_INVALID_VALUE\n");
                    break;
                case CL_DEVICE_NOT_FOUND:
                    printf("CL_DEVICE_NOT_FOUND\n");
                    break;
                case CL_OUT_OF_RESOURCES:
                    printf("CL_OUT_OF_RESOURCES\n");
                    break;
                case CL_OUT_OF_HOST_MEMORY:
                    printf("CL_OUT_OF_HOST_MEMORY\n");
                    break;
            }
            printf("clGetDeviceIDs error %d\n", clErr); exit(1); }
        printf("%d devices found\n", deviceCount);
        if(devID >= deviceCount){ printf("Wrong device selected: %d!\n", devID); exit(1); }
        // запрашиваем список девайсов у выбранной платформы
        cl_device_id *deviceList = new cl_device_id[deviceCount]; // list of devices
        clErr = clGetDeviceIDs(platformList[platform_id], CL_DEVICE_TYPE_ALL,
                               (cl_uint)deviceCount, deviceList, NULL);
        if(clErr != CL_SUCCESS){ printf("clGetDeviceIDs error %d\n", clErr); exit(1); }
        delete[] platformList; // больше не нужно
        // печатаем девайсы платформы
        for(int i=0; i<deviceCount; i++){
            clErr = clGetDeviceInfo(deviceList[i], CL_DEVICE_NAME, STR_SIZE, nameBuf, 0);
            if(clErr != CL_SUCCESS){ printf("clGetDeviceInfo error %d\n", clErr); exit(1); }
            printf(" Device %d: %s \n", i, nameBuf);
        }

        // CONTEXT
        clContext = clCreateContext( NULL, 1, &deviceList[devID], 0, 0, &clErr);
        if(clErr != CL_SUCCESS){ printf("clCreateContext error %d\n",clErr ); exit(1); }
        // COMMAND QUEUE
        clQueue = clCreateCommandQueue(clContext, deviceList[devID], 0, &clErr);
        if(clErr != CL_SUCCESS){ printf("clCreateCommandQueue %d\n",clErr ); exit(1); }
        // PROGRAM
        const char *cPathAndName="kernel.cl"; // файл с исходным кодом кернелов
        printf("Loading program from %s\n", cPathAndName);

        // сюда можно напихать каких нужно дефайнов, чтобы они подставились в программу
        const char *cDefines=" /* add your defines */ ";

        char * cSourceCL = NULL; // буфер для исходного кода
        { // читаем файл
        FILE *f=fopen(cPathAndName, "rb");
        if(!f){ printf("Can't open program file %s!\n", cPathAndName); exit(1); }
        fseek(f, 0, SEEK_END); // считаем размер
        size_t fileSize = ftell(f);
        rewind(f);
        int codeSize = fileSize + strlen(cDefines); // считаем общий размер: код + дефайны
        cSourceCL = new char [codeSize + 1/*zero-terminated*/]; // выделяем буфер
        memcpy(cSourceCL,cDefines,strlen(cDefines)); // подставляем дефайны
        size_t nd = fread(cSourceCL+strlen(cDefines),1,fileSize,f); // читаем
        if(nd != fileSize){ printf("Failed to read program %s!\n", cPathAndName); exit(1); }
        cSourceCL[codeSize]=0; // заканчиваем строку нулем!
        }
        if(cSourceCL == NULL){printf("Can't get program from %s!\n", cPathAndName); exit(1); }
        // сдаем исходники в OpenCL
        size_t szKernelLength = strlen(cSourceCL);
        clProgram = clCreateProgramWithSource(clContext, 1, (const char **) &cSourceCL,
        &szKernelLength, &clErr);
        if(clErr != CL_SUCCESS){printf("clCreateProgramWithSource error %d\n",clErr ); exit(1);}

        // компилим кернел-программу
        printf("clBuildProgram... ");
        clErr = clBuildProgram(clProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);
        printf("done\n");

        // запрашиваем размер лога компиляции
        int LOG_S=0;
        clErr = clGetProgramBuildInfo(clProgram, deviceList[devID], CL_PROGRAM_BUILD_LOG,
        0, NULL, (size_t*)&LOG_S);

        if(clErr != CL_SUCCESS){ printf("clGetProgramBuildInfo error %d\n", clErr);exit(1); }
        if(LOG_S>8){ // если там не пусто - печатаем лог
        char *programLog= new char[LOG_S];
        clErr = clGetProgramBuildInfo(clProgram, deviceList[devID], CL_PROGRAM_BUILD_LOG,
        LOG_S, programLog, 0);
        if(clErr != CL_SUCCESS){ printf("clGetProgramBuildInfo error %d\n", clErr);exit(1); }
        printf("%s\n", programLog);
        delete[] programLog;
        }
        if(clErr != CL_SUCCESS){ printf("Compilation failed with error: %d\n",clErr); exit(1); }
        delete [] cSourceCL;
    }
}

int solve_with_ocl  ( const CSRPortrait &portrait_A,
                      const std::vector <double> &A_value,
                      const std::vector <double> &b,
                      const double eps,
                      const int maxit,
                      std::vector <double> &x
                      //~ std::vector <double> &times_of_kernels
                      ) {
    CSRPortrait portrait_M;
    std::vector <double> M_value;
    construct_matrix_M(portrait_A, A_value, portrait_M, M_value);
    double rho_k = 0, rho_k_1 = 0, eps_q = eps*eps;
    std::vector <double> r(b.begin(), b.end()); //    𝒓0 = 𝒃

    int N = b.size(); // размер вектора
    
    x.clear();
    x.resize(N, 0);  //    𝒙0 = 0
    std::vector <double> z(portrait_A.N);
    std::vector <double> p(portrait_A.N);
    std::vector <double> q(portrait_A.N);
    
    // Инициализация OpenCL
    const cl_int devID = 0; // Номер нужного девайса
    const char *platformName = "AMD Accelerated Parallel Processing"; //Нужная платформа // "Intel(R) OpenCL" // "NVIDIA CUDA"
    // OpenCL переменные
    cl_context clContext; // OpenCL контекст
    cl_command_queue clQueue; // OpenCL очередь команд
    cl_program clProgram; // OpenCL программа
    cl_int clErr; // код возврата из OpenCL функций
    ocl_init(devID, platformName, clContext, clQueue, clProgram, clErr);

    // KERNELS
    printf("Creating kernels\n");
    cl_kernel knlSPMV_CSR; // SPMV_CSR
    knlSPMV_CSR = clCreateKernel(clProgram, "knlSPMV_CSR", &clErr); //что если сделать 2 экземпляра одного кернела?
    if(clErr != CL_SUCCESS){ printf("clCreateKernel knlSPMV_CSR error: %d\n",clErr); exit(1); }

    cl_kernel knlDOT;
    knlDOT = clCreateKernel(clProgram, "knlDOT", &clErr); // создаем кернел
    if(clErr != CL_SUCCESS){ printf("clCreateKernel knlSUM error: %d\n",clErr); exit(1); }

    cl_kernel knlAXPBY; // x=ax+y
    knlAXPBY = clCreateKernel(clProgram, "knlAXPBY", &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateKernel knlAXPBY error: %d\n",clErr); exit(1); }

    // BUFFERS...
    printf("Creating opencl buffers\n");
    // ...для матрицы предобуславливателя
    cl_mem clM = clCreateBuffer(clContext, CL_MEM_READ_WRITE, M_value.size()*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clM error %d\n",clErr); exit(1); }

    cl_mem clIM = clCreateBuffer(clContext, CL_MEM_READ_WRITE, portrait_M.cumulative_sum.size()*sizeof(int),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clIM error %d\n",clErr); exit(1); }

    cl_mem clJM = clCreateBuffer(clContext, CL_MEM_READ_WRITE, portrait_M.column_numbers.size()*sizeof(int),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clJM error %d\n",clErr); exit(1); }
    // ...для матрицы A
    cl_mem clA = clCreateBuffer(clContext, CL_MEM_READ_WRITE, A_value.size()*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clA error %d\n",clErr); exit(1); }

    cl_mem clIA = clCreateBuffer(clContext, CL_MEM_READ_WRITE, portrait_A.cumulative_sum.size()*sizeof(int),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clIA error %d\n",clErr); exit(1); }

    cl_mem clJA = clCreateBuffer(clContext, CL_MEM_READ_WRITE, portrait_A.column_numbers.size()*sizeof(int),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clJA error %d\n",clErr); exit(1); }
    // ...вектора r
    cl_mem clR = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clR error %d\n",clErr); exit(1); }
    // ...вектора z
    cl_mem clZ = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clZ error %d\n",clErr); exit(1); }
    // ...вектора p
    cl_mem clP = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clP error %d\n",clErr); exit(1); }
    // ...вектора q
    cl_mem clQ = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clQ error %d\n",clErr); exit(1); }
    // ...вектор x
    cl_mem clX = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clX error %d\n",clErr); exit(1); }    
    
    // ...для результата частичного скалярного произведения
    // ...сперва посчитаем размер этого буффера
    #define REDUCTION_LWS 256 // размер рабочей группы для суммы
    #define REDUCTION_ITEM 8 // сколько элементов будет суммировать один work-item (нить)
    // ...размер буфера частичных сумм рабочих групп
    int REDUCTION_BUFSIZE = ((N/REDUCTION_ITEM)/REDUCTION_LWS) + ((N/REDUCTION_ITEM)%REDUCTION_LWS>0);
    // ...буфер под частичные суммы рабочих групп
    cl_mem clSum = clCreateBuffer(clContext, CL_MEM_READ_WRITE, REDUCTION_BUFSIZE*sizeof(double), NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clSum error %d\n",clErr); exit(1); }
    // ...сразу выделим для частичных сумм место на хосте
    std::vector <double> Sum(REDUCTION_BUFSIZE, 0);
        
    //~ cl_mem clY = clCreateBuffer(clContext, CL_MEM_READ_WRITE, portrait_A.M*sizeof(double),NULL, &clErr);
    //~ if(clErr != CL_SUCCESS){ printf("clCreateBuffer clY error %d\n",clErr); exit(1); }
    printf("Init done\n");

    std::vector <double> y(portrait_A.M, 0);

    // копируем вектора на девайс
    clErr = clEnqueueWriteBuffer(clQueue, clX, CL_TRUE, 0, N*sizeof(double), &x[0], 0,NULL,NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clX error %d\n", clErr); exit(1); }

    clErr = clEnqueueWriteBuffer(clQueue, clR, CL_TRUE, 0, N*sizeof(double), &r[0], 0,NULL,NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clX error %d\n", clErr); exit(1); }

    clErr = clEnqueueWriteBuffer(clQueue, clM, CL_TRUE, 0, M_value.size()*sizeof(double), &M_value[0], 0,NULL,NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clM error %d\n", clErr); exit(1); }

    clErr = clEnqueueWriteBuffer(clQueue, clIM, CL_TRUE, 0, portrait_M.cumulative_sum.size()*sizeof(int), &portrait_M.cumulative_sum[0], 0,NULL,NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clIM error %d\n", clErr); exit(1); }

    clErr = clEnqueueWriteBuffer(clQueue, clJM, CL_TRUE, 0, portrait_M.column_numbers.size()*sizeof(int), &portrait_M.column_numbers[0], 0,NULL,NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clJM error %d\n", clErr); exit(1); }

    int k = 0;                  //    𝑘 = 0
    do {
        ++k;
        //~ double start = omp_get_wtime();
        // Запуск kernelSPMV с предобуславливателем...
        // ...выставляем параметры запуска
        size_t lws = 128; // размер рабочей группы
        size_t gws = portrait_M.M; // общее число заданий
        if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
        // ...выставляем аргументы кернелу 
        //  (Что если вынести инициализацию неизменных компонент?)
        clSetKernelArg(knlSPMV_CSR, 0, sizeof(int), &portrait_M.M);
        clSetKernelArg(knlSPMV_CSR, 1, sizeof(cl_mem), &clM);
        clSetKernelArg(knlSPMV_CSR, 2, sizeof(cl_mem), &clIM);
        clSetKernelArg(knlSPMV_CSR, 3, sizeof(cl_mem), &clJM);
        clSetKernelArg(knlSPMV_CSR, 4, sizeof(cl_mem), &clR);
        clSetKernelArg(knlSPMV_CSR, 5, sizeof(cl_mem), &clZ);
        // ...отправляем на исполнение
        clErr= clEnqueueNDRangeKernel(clQueue, knlSPMV_CSR, 1, NULL, &gws, &lws, 0, NULL, NULL);
        if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
        clFinish(clQueue); // ждем завершения
        //~ SpMV(portrait_M, M_value, r, z); //𝒛𝑘 = 𝑴−1𝒓𝑘−1 // SpMV
        //~ double stop = omp_get_wtime();
        //~ times_of_kernels[0] += stop - start;
        //~ start = omp_get_wtime();
        
        // Запуск кернела DOT...
        // ...выставляем параметры запуска
        lws = REDUCTION_LWS; // размер рабочей группы
        gws = (N/REDUCTION_ITEM); // общее число заданий
        if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
        // ...выставляем аргументы кернелу
        clSetKernelArg(knlDOT, 0, sizeof(int), &N);
        clSetKernelArg(knlDOT, 1, sizeof(cl_mem), &clR);
        clSetKernelArg(knlDOT, 2, sizeof(cl_mem), &clZ);
        clSetKernelArg(knlDOT, 3, sizeof(cl_mem), &clSum);  
        // ...отправляем на исполнение
        clErr= clEnqueueNDRangeKernel(clQueue, knlDOT, 1, NULL, &gws, &lws, 0, NULL, NULL);
        if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
        clFinish(clQueue); // ждем завершения
        // ...забираем результат с девайса
        clErr = clEnqueueReadBuffer(clQueue, clSum, CL_TRUE, 0, REDUCTION_BUFSIZE*sizeof(double), &Sum[0], 0, NULL, NULL);
        if(clErr != CL_SUCCESS){printf("clEnqueueReadBuffer clSum error %d\n", clErr); exit(1);}
        clFinish(clQueue);
        // ...досуммируем результат на хосте, там осталось где-то 0.05% от общего объема работы
        rho_k = 0.0;
        #pragma omp parallel for reduction(+:rho_k)
        for(int i=0; i<REDUCTION_BUFSIZE; ++i) rho_k += Sum[i];
        // досуммируем по всей MPI группе, если у нас имеется MPI распараллеливание
        //~ double gsum;
        //~ MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        //~ rho_k = dot(r, z);               //𝜌𝑘 = (𝒓𝑘−1, 𝒛𝑘) // dot
        //~ stop = omp_get_wtime();
        //~ times_of_kernels[1] += stop - start;
        double a, b;
        if (k == 1) {                    //if 𝑘 = 1 then
            // Для копирования вектора используем кернел AXPBY
            lws = 128; // размер рабочей группы
            gws = N; // общее число заданий
            if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
            // ...выставляем аргументы кернелу
            clSetKernelArg(knlAXPBY, 0, sizeof(int), &N);
            clSetKernelArg(knlAXPBY, 1, sizeof(cl_mem), &clZ);
            clSetKernelArg(knlAXPBY, 2, sizeof(cl_mem), &clZ);
            clSetKernelArg(knlAXPBY, 3, sizeof(cl_mem), &clP);
            a = 1.;
            clSetKernelArg(knlAXPBY, 4, sizeof(double), &a);
            b = 0.;
            clSetKernelArg(knlAXPBY, 5, sizeof(double), &b);
            // ...отправляем на исполнение
            clErr= clEnqueueNDRangeKernel(clQueue, knlAXPBY, 1, NULL, &gws, &lws, 0, NULL, NULL);
            if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
            clFinish(clQueue); // ждем завершения
            //~ p = z;                       //𝒑𝑘 = 𝒛𝑘
        } else {
            double beta = rho_k / rho_k_1; //𝛽𝑘 = 𝜌𝑘/𝜌𝑘−1
            //~ start = omp_get_wtime();
            //Запускаем кернел AXPBY
            lws = 128; // размер рабочей группы
            gws = N; // общее число заданий
            if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
            // ...выставляем аргументы кернелу
            clSetKernelArg(knlAXPBY, 0, sizeof(int), &N);
            clSetKernelArg(knlAXPBY, 1, sizeof(cl_mem), &clZ);
            clSetKernelArg(knlAXPBY, 2, sizeof(cl_mem), &clP);
            clSetKernelArg(knlAXPBY, 3, sizeof(cl_mem), &clP);
            a = 1.;
            clSetKernelArg(knlAXPBY, 4, sizeof(double), &a);
            clSetKernelArg(knlAXPBY, 5, sizeof(double), &beta);
            // ...отправляем на исполнение
            clErr= clEnqueueNDRangeKernel(clQueue, knlAXPBY, 1, NULL, &gws, &lws, 0, NULL, NULL);
            if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
            clFinish(clQueue); // ждем завершения            
            //~ axpy(z, p, beta, p);         //𝒑𝑘 = 𝒛𝑘 + 𝛽𝑘𝒑𝑘−1 // axpy
            //~ stop = omp_get_wtime();
            //~ times_of_kernels[2] += stop - start;
        }
        //~ start = omp_get_wtime();
        // Запуск кернела SPMV с матрицей A
        lws = 128; // размер рабочей группы
        gws = portrait_M.M; // общее число заданий
        if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
        // ...выставляем аргументы кернелу 
        //  (Что если вынести инициализацию неизменных компонент?)
        clSetKernelArg(knlSPMV_CSR, 0, sizeof(int), &portrait_A.M);
        clSetKernelArg(knlSPMV_CSR, 1, sizeof(cl_mem), &clA);
        clSetKernelArg(knlSPMV_CSR, 2, sizeof(cl_mem), &clIA);
        clSetKernelArg(knlSPMV_CSR, 3, sizeof(cl_mem), &clJA);
        clSetKernelArg(knlSPMV_CSR, 4, sizeof(cl_mem), &clP);
        clSetKernelArg(knlSPMV_CSR, 5, sizeof(cl_mem), &clQ);
        // ...отправляем на исполнение
        clErr= clEnqueueNDRangeKernel(clQueue, knlSPMV_CSR, 1, NULL, &gws, &lws, 0, NULL, NULL);
        if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
        clFinish(clQueue); // ждем завершения
        //~ SpMV(portrait_A, A_value, p, q); //𝒒𝑘 = 𝑨𝒑𝑘 // SpMV
        //~ stop = omp_get_wtime();
        //~ times_of_kernels[0] += stop - start;
        //~ start = omp_get_wtime();
        
        // Запуск кернела DOT...
        // ...выставляем параметры запуска
        lws = REDUCTION_LWS; // размер рабочей группы
        gws = (N/REDUCTION_ITEM); // общее число заданий
        if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
        // ...выставляем аргументы кернелу
        clSetKernelArg(knlDOT, 0, sizeof(int), &N);
        clSetKernelArg(knlDOT, 1, sizeof(cl_mem), &clP);
        clSetKernelArg(knlDOT, 2, sizeof(cl_mem), &clQ);
        clSetKernelArg(knlDOT, 3, sizeof(cl_mem), &clSum);  
        // ...отправляем на исполнение
        clErr= clEnqueueNDRangeKernel(clQueue, knlDOT, 1, NULL, &gws, &lws, 0, NULL, NULL);
        if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
        clFinish(clQueue); // ждем завершения
        // ...забираем результат с девайса
        clErr = clEnqueueReadBuffer(clQueue, clSum, CL_TRUE, 0, REDUCTION_BUFSIZE*sizeof(double), &Sum[0], 0, NULL, NULL);
        if(clErr != CL_SUCCESS){printf("clEnqueueReadBuffer clSum error %d\n", clErr); exit(1);}
        clFinish(clQueue);
        // ...досуммируем результат на хосте, там осталось где-то 0.05% от общего объема работы
        double dot_p_q = 0.0;
        #pragma omp parallel for reduction(+:dot_p_q)
        for(int i=0; i<REDUCTION_BUFSIZE; ++i) dot_p_q += Sum[i];        
        double alpha = rho_k / dot_p_q;  //𝛼𝑘 = 𝜌𝑘/(𝒑𝑘, 𝒒𝑘) // dot
        //~ stop = omp_get_wtime();
        //~ times_of_kernels[1] += stop - start;
        //~ start = omp_get_wtime();

        //Запускаем кернел AXPBY
        lws = 128; // размер рабочей группы
        gws = N; // общее число заданий
        if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
        // ...выставляем аргументы кернелу
        clSetKernelArg(knlAXPBY, 0, sizeof(int), &N);
        clSetKernelArg(knlAXPBY, 1, sizeof(cl_mem), &clX);
        clSetKernelArg(knlAXPBY, 2, sizeof(cl_mem), &clP);
        clSetKernelArg(knlAXPBY, 3, sizeof(cl_mem), &clX);
        a = 1.;
        clSetKernelArg(knlAXPBY, 4, sizeof(double), &a);
        clSetKernelArg(knlAXPBY, 5, sizeof(double), &alpha);
        // ...отправляем на исполнение
        clErr= clEnqueueNDRangeKernel(clQueue, knlAXPBY, 1, NULL, &gws, &lws, 0, NULL, NULL);
        if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
        clFinish(clQueue); // ждем завершения        
        // ...выставляем аргументы кернелу
        clSetKernelArg(knlAXPBY, 0, sizeof(int), &N);
        clSetKernelArg(knlAXPBY, 1, sizeof(cl_mem), &clR);
        clSetKernelArg(knlAXPBY, 2, sizeof(cl_mem), &clQ);
        clSetKernelArg(knlAXPBY, 3, sizeof(cl_mem), &clR);
        a = 1.;
        clSetKernelArg(knlAXPBY, 4, sizeof(double), &a);
        b = -alpha;
        clSetKernelArg(knlAXPBY, 5, sizeof(double), &b);
        // ...отправляем на исполнение
        clErr= clEnqueueNDRangeKernel(clQueue, knlAXPBY, 1, NULL, &gws, &lws, 0, NULL, NULL);
        if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
        clFinish(clQueue); // ждем завершения                
        //~ axpy(x, p, alpha, x);            //𝒙𝑘 = 𝒙𝑘−1 + 𝛼𝑘𝒑𝑘 // axpy
        //~ axpy(r, q, alpha*(-1), r);       //𝒓𝑘 = 𝒓𝑘−1 − 𝛼𝑘𝒒𝑘 // axpy
        //~ stop = omp_get_wtime();
        //~ times_of_kernels[2] += stop - start;
        rho_k_1 = rho_k;
        if(k%10 == 0) {
            std::cout << "Iteration: " << k << ", rho: " << rho_k << std::endl; 
        }

    } while (rho_k > eps_q && k < maxit);//𝜌𝑘 > 𝜀^2 and k < maxit
    // Забираем x с девайса
    clErr = clEnqueueReadBuffer(clQueue, clX, CL_TRUE, 0, N*sizeof(double), &x[0], 0, NULL, NULL);
    if(clErr != CL_SUCCESS){printf("clEnqueueReadBuffer clSum error %d\n", clErr); exit(1);}
    clFinish(clQueue);  
    //~ times_of_kernels[0] /= 2*k;
    //~ times_of_kernels[1] /= 2*k;
    //~ times_of_kernels[2] /= 3*k;
    return k;
}


int args_parsing(int argc, char *argv[], std::vector <int> &args_values, double &epsilon) {
    args_values.clear();

    int Nx = 0, Ny = 0, k1 = 0, k2 = 0;
    double eps = 0.;
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
                if(num_threads < 1 || num_threads >= MAX_SIZE) {
                    std::cerr << "-t value most be greater then 1 and less than " << MAX_SIZE << std::endl;
                    std::cerr << "-t value will be default " << std::endl;
                } else {
                    omp_set_num_threads(num_threads);
                }
            }
        } else if(arg == "-e" || "--epsilon") {
            if (i + 1 < argc) {
                eps = strtod(argv[++i], nullptr);
                if(eps <= 0.) {
                    std::cerr << "-e value most be greater then 0." <<std::endl;
                    std::cerr << "-e value will be default" <<std::endl;
                    eps = 1e-6;
                }
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
    epsilon = eps;

    return 0;
}

int main(int argc, char *argv[]) {

    std::vector <int> args_values;
    double eps = 1e-6;
    args_parsing(argc, argv, args_values, eps);
    int Nx = args_values[0], Ny = args_values[1], k1 = args_values[2], k2 = args_values[3];
    bool debug_mode = args_values[4];

    CSRPortrait newcsr, tcsr;
    int occupied_memory = 0;
    double start = omp_get_wtime();
    occupied_memory = generate_grid(Nx, Ny, k1, k2, newcsr);
    double stop = omp_get_wtime();
    std::cout << "Grid generation time:            " << stop - start << std::endl;
    std::cout << "Used memory for grid generation: " << occupied_memory << " bytes" << std::endl;

    start = omp_get_wtime();
    occupied_memory = construct_matrix_E_f_E(newcsr, tcsr);
    stop = omp_get_wtime();
    std::cout << "EfE matrix generation time:      " << stop - start << std::endl;
    std::cout << "Used memory for EfE generation:  " << occupied_memory << " bytes" << std::endl;

    std::vector <double> A_value;
    start = omp_get_wtime();
    initialization_A(tcsr, A_value);
    stop = omp_get_wtime();
    std::cout << "Matrix values generation time:   " << stop - start << std::endl;

    std::vector <double> b;
    start = omp_get_wtime();
    fill_in_b(b, tcsr.M);
    stop = omp_get_wtime();
    std::cout << "Vector b values generation time: " << stop - start << std::endl;

    std::vector <double> x;
    std::vector <double> times_of_kernels = {0, 0, 0};
    int maxit = 500;
    start = omp_get_wtime();
    auto iter_count = solve(tcsr, A_value, b, eps, maxit, x, times_of_kernels, debug_mode);
    stop = omp_get_wtime();
    std::cout << "Solve time:                      " << stop - start << std::endl;
    std::cout << "number of iterations:            " << iter_count << std::endl;
    std::cout << "Average working time SpMV kernel:" << times_of_kernels[0] << std::endl;
    std::cout << "Average working time dot kernel: " << times_of_kernels[1] << std::endl;
    std::cout << "Average working time axpy kernel:" << times_of_kernels[2] << std::endl;

    if(debug_mode) {
        std::cout << "Matrix A:" << std::endl;
        print_CSRPortrait(newcsr);
        print_CSRPortrait(tcsr);
        for(auto x : A_value) {
            std::cout << x << ' ';
        }
        std::cout << std::endl;

        std::cout << "Vector b:" << std::endl;
        for(auto x : b) {
            std::cout << x << ' ';
        }
        std::cout << std::endl;

        std::cout << "Vector x:" << std::endl;
        for(auto x : x) {
            std::cout << x << ' ';
        }
        std::cout << std::endl;
    }
    double A_sum = 0, x_sum = 0;
    for(auto x: A_value) {
        A_sum += x;
    }
    for(auto x: x) {
        x_sum += x;
    }
    std::vector <double> z(x.size());
    std::vector <double> r(x.size());
    SpMV(tcsr, A_value, x, z);
    axpy(z, b, -1, r);
    auto square_of_discrepancy = dot(r,r);
    std::cout << "Epsilon:                         " << eps << std::endl;
    std::cout << "The square of the discrepancy:   " << square_of_discrepancy << std::endl;
    std::cout << "Sum elements of A                " << A_sum << std::endl;
    std::cout << "Sum elements of x                " << x_sum << std::endl;
    
    std::vector<double> ocl_x;
    iter_count = solve_with_ocl(tcsr, A_value, b, eps, maxit, ocl_x);
    
    // сравниваем с хостом, находим норму отличий
    double sum=0;
    for(int i=0; i<x.size(); i++){
        sum += fabs(x[i]-ocl_x[i]);
    }
    printf("Test execution SPMV done\n Error = %g\n", sum);

    return 0;
}
