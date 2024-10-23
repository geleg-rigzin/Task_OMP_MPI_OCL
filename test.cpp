//~ // C standard includes
//~ #include <stdio.h>

//~ // OpenCL includes
//~ #include <CL/cl.h>

//~ int main()
//~ {
    //~ cl_int CL_err = CL_SUCCESS;
    //~ cl_uint numPlatforms = 0;

    //~ CL_err = clGetPlatformIDs( 0, NULL, &numPlatforms );

    //~ if (CL_err == CL_SUCCESS)
        //~ printf("%u platform(s) found\n", numPlatforms);
    //~ else
        //~ printf("clGetPlatformIDs(%i)\n", CL_err);

    //~ return 0;
//~ }

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h> // OpenCL
int main(int /*argc*/, char** /*argv*/){

    // Входные параметры
    const cl_int devID = 0; // Номер нужного девайса
    const char *platformName = "AMD Accelerated Parallel Processing"; //Нужная платформа // "Intel(R) OpenCL" // "NVIDIA CUDA"
    // OpenCL переменные
    cl_context clContext; // OpenCL контекст
    cl_command_queue clQueue; // OpenCL очередь команд
    cl_program clProgram; // OpenCL программа
    cl_int clErr; // код возврата из OpenCL функций

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
    // KERNELS
    printf("Creating kernels\n");
    cl_kernel knlAXPBY; // x=ax+y
    knlAXPBY = clCreateKernel(clProgram, "knlAXPBY", &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateKernel knlAXPBY error: %d\n",clErr); exit(1); }
    printf(" knlAXPBY\n");
    const int N=10000123; // размер тестовых векторов

    // BUFFERS
    printf("Creating opencl buffers\n");
    cl_mem clX = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clX error %d\n",clErr); exit(1); }
    cl_mem clY = clCreateBuffer(clContext, CL_MEM_READ_WRITE, N*sizeof(double),NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clY error %d\n",clErr); exit(1); }
    printf("Init done\n");

    // TEST EXECUTION
    // данные на стороне CPU
    double *X = new double[N];
    double *Y = new double[N];
    const double a = 1.234, b = 3.456;
    // заполняем вектора какой-то ерундой
    for(int i=0; i<N; i++){
    X[i]=(double)(i%123)*(i%456);
    Y[i]=(double)(i%248)*(i%134);
    }
    // копируем вектора на девайс
    clErr = clEnqueueWriteBuffer(clQueue, clX, CL_TRUE, 0, N*sizeof(double), X, 0,NULL,NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clX error %d\n", clErr); exit(1); }
    clErr = clEnqueueWriteBuffer(clQueue, clY, CL_TRUE, 0, N*sizeof(double), Y, 0,NULL,NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clY error %d\n", clErr); exit(1); }
    // выставляем параметры запуска
    size_t lws = 128; // размер рабочей группы
    size_t gws = N; // общее число заданий
    if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
    // выставляем аргументы кернелу
    clSetKernelArg(knlAXPBY, 0, sizeof(int), &N);
    clSetKernelArg(knlAXPBY, 1, sizeof(cl_mem), &clX);
    clSetKernelArg(knlAXPBY, 2, sizeof(cl_mem), &clY);
    clSetKernelArg(knlAXPBY, 3, sizeof(double), &a);
    clSetKernelArg(knlAXPBY, 4, sizeof(double), &b);
    // отправляем на исполнение
    clErr= clEnqueueNDRangeKernel(clQueue, knlAXPBY, 1, NULL, &gws, &lws, 0, NULL, NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
    clFinish(clQueue); // ждем завершения

    // делаем ту же операцию на хосте
    for(int i=0; i<N; ++i){
        X[i] = a*X[i] + b*Y[i];
    }
    // забираем результат с девайса
    double *R = new double[N];
    clErr = clEnqueueReadBuffer(clQueue, clX, CL_TRUE, 0, N*sizeof(double), R, 0, NULL, NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueReadBuffer clX error %d\n", clErr); exit(1); }
    // сравниваем с хостом, находим норму отличий
    double sum=0;
    for(int i=0; i<N; i++){
        sum += fabs(R[i]-X[i]);
    }
    printf("Test execution AXPBY done\n Error = %g\n", sum/N);
    
    cl_kernel knlDOT;
    knlDOT = clCreateKernel(clProgram, "knlDOT", &clErr); // создаем кернел
    if(clErr != CL_SUCCESS){ printf("clCreateKernel knlSUM error: %d\n",clErr); exit(1); }
    #define REDUCTION_LWS 256 // размер рабочей группы для суммы
    #define REDUCTION_ITEM 8 // сколько элементов будет суммировать один work-item (нить)
    // размер буфера частичных сумм рабочих групп
    int REDUCTION_BUFSIZE = ((N/REDUCTION_ITEM)/REDUCTION_LWS) + ((N/REDUCTION_ITEM)%REDUCTION_LWS>0);
    // буфер под частичные суммы рабочих групп
    cl_mem clSum = clCreateBuffer(clContext, CL_MEM_READ_WRITE,
                                  REDUCTION_BUFSIZE*sizeof(double), NULL, &clErr);
    if(clErr != CL_SUCCESS){ printf("clCreateBuffer clSum error %d\n",clErr); exit(1); }
    // выставляем параметры запуска
    lws = REDUCTION_LWS; // размер рабочей группы
    gws = (N/REDUCTION_ITEM); // общее число заданий
    if(gws%lws>0) gws += lws-gws%lws; // делаем кратное lws
    //еще раз отправляем X, на устройстве он перезаписан предыдущим кернелом
    // копируем вектора на девайс
    clErr = clEnqueueWriteBuffer(clQueue, clX, CL_TRUE, 0, N*sizeof(double), X, 0,NULL,NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clX error %d\n", clErr); exit(1); }
    clErr = clEnqueueWriteBuffer(clQueue, clY, CL_TRUE, 0, N*sizeof(double), Y, 0,NULL,NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueWriteBuffer clY error %d\n", clErr); exit(1); }
    // выставляем аргументы кернелу
    clSetKernelArg(knlDOT, 0, sizeof(int), &N);
    clSetKernelArg(knlDOT, 1, sizeof(cl_mem), &clX);
    clSetKernelArg(knlDOT, 2, sizeof(cl_mem), &clY);
    clSetKernelArg(knlDOT, 3, sizeof(cl_mem), &clSum);  
    // отправляем на исполнение
    clErr= clEnqueueNDRangeKernel(clQueue, knlDOT, 1, NULL, &gws, &lws, 0, NULL, NULL);
    if(clErr != CL_SUCCESS){ printf("clEnqueueNDRangeKernel error %d\n",clErr); exit(1); }
    clFinish(clQueue); // ждем завершения
    // забираем результат с девайса
    double *Sum = new double[REDUCTION_BUFSIZE];
    clErr = clEnqueueReadBuffer(clQueue, clSum, CL_TRUE, 0,
                                REDUCTION_BUFSIZE*sizeof(double), Sum, 0, NULL, NULL);
    if(clErr != CL_SUCCESS){printf("clEnqueueReadBuffer clSum error %d\n", clErr); exit(1);}
    clFinish(clQueue);
    // досуммируем результат на хосте, там осталось где-то 0.05% от общего объема работы
    double lsum = 0.0;
    #pragma omp parallel for reduction(+:lsum)
    for(int i=0; i<REDUCTION_BUFSIZE; ++i) lsum += Sum[i];
    // досуммируем по всей MPI группе, если у нас имеется MPI распараллеливание
    //~ double gsum;
    //~ MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double hdot;
    #pragma omp parallel for reduction(+:hdot)
    for(int i=0; i<N; ++i) hdot += X[i]*Y[i];
    printf("Test execution DOT done\n Error = %g\n", fabs(hdot-lsum));
    printf("host   = %g\n", hdot);
    printf("device = %g\n", lsum);
    
    delete [] R; delete [] X; delete [] Y;
    clErr = clReleaseMemObject(clX);
    if(clErr != CL_SUCCESS){ printf("clReleaseMemObject clX error %d\n", clErr); exit(1); }
    clErr = clReleaseMemObject(clY);
    if(clErr != CL_SUCCESS){ printf("clReleaseMemObject clY error %d\n", clErr); exit(1); }

}

