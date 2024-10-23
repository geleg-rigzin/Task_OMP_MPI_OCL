#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <omp.h>
#include <cmath>
//~ #include <climits>
#include "mpi.h"
#define MAX_SIZE 1000000
#include <cassert>
// -----------------------------------------------------------------------------------------------------------
// Global data
// -----------------------------------------------------------------------------------------------------------
int mpi_initialized = 0; // MPI initialization flag
int MyID = 0; // process ID
int NumProc = 1; // number of processes
int MASTER_ID = 0; // master process ID
MPI_Comm MCW = MPI_COMM_WORLD; // default communicator

struct CSRPortrait {
    int N_own = 0;
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

struct tCommScheme {
    int NumOfNeighbours = 0;
    std::vector <int> SendList;
    std::vector <int> RecvList;
    std::vector <int> SendOffset;
    std::vector <int> RecvOffset;
    std::vector <int> ListOfNeighbours;
    tCommScheme(int N, const CSRPortrait &Matrix, std::vector<int> &Part, std::vector<int> &L2G) {
        int N_own = Matrix.N_own;
        if(N - N_own) {//halo exist
            int prev_neighbour = Part[Matrix.N_own];
            ListOfNeighbours.push_back(prev_neighbour);
            RecvOffset.push_back(0);
            auto i = N_own;
            for(; i < N; ++i) {
                RecvList.push_back(i);//don't need sort
                if(prev_neighbour != Part[i]) {
                    prev_neighbour = Part[i];
                    ListOfNeighbours.push_back(prev_neighbour);
                    RecvOffset.push_back(i - N_own);
                }
            }
            RecvOffset.push_back(i - N_own);
        }
        //~ all_proc[MyID] = 1;

        NumOfNeighbours = ListOfNeighbours.size();
        std::vector <int> all_proc(NumProc, -1);
        for(int i = 0; i < NumOfNeighbours; ++i) {
            all_proc[ListOfNeighbours[i]] = i;
        }
        std::vector <std::vector <int>> send_lists(NumOfNeighbours);
        for(auto i = 0; i < Matrix.M; ++i) {
            for(auto k = Matrix.cumulative_sum[i]; k < Matrix.cumulative_sum[i+1]; ++k) {
                auto vertex = Matrix.column_numbers[k];
                if(vertex >= N_own) {
                    if(!send_lists[all_proc[Part[vertex]]].empty()) {
                        if(send_lists[all_proc[Part[vertex]]].back() == i) {
                            continue;
                        }
                    }
                    send_lists[all_proc[Part[vertex]]].push_back(i);//already sorted
                }
            }
        }
        SendOffset.push_back(0);
        for(auto i = 0; i < NumOfNeighbours; ++i) {
            SendOffset.push_back(send_lists[i].size() + SendOffset.back());
            for(auto element : send_lists[i]) {
                SendList.push_back(element);
            }
        }
    }

    const int GetNumOfNeighbours() {return NumOfNeighbours;} // число соседей
    std::vector <int>& GetSendList() {return SendList;} // список ячеек на отправку по всем соседям
    std::vector <int>& GetRecvList() {return RecvList;} // список ячеек на прием по всем соседям
    std::vector <int>& GetSendOffset() {return SendOffset;} // смещения списков по каждому соседу на отправку
    std::vector <int>& GetRecvOffset() {return RecvOffset;} // смещения списков по каждому соседу на прием
    std::vector <int>& GetListOfNeighbours() {return ListOfNeighbours;}  // номера процессов соседей
    //MPI_Comm GetMyComm(); // коммуникатор для данной группы (MPI_COMM_WORLD)
};

int generate_grid(int Nx,
                  int Ny,
                  int k1,
                  int k2,
                  CSRPortrait &EN,
                  int Px,
                  int Py,
                  std::vector <int> &L2G,
                  std::vector <int> &Part) {
    auto sum_k1_k2 = k1 + k2;
    auto prev = 0;
    EN.column_numbers.clear();
    EN.cumulative_sum.clear();
    L2G.clear();
    Part.clear();

    EN.cumulative_sum.push_back(prev);

    auto MyY = MyID / Px;
    auto MyX = MyID % Px;

    auto ibeg = MyY*((Ny - 1)/Py) + ( MyY < (Ny - 1)%Py ? MyY : (Ny - 1)%Py );
    auto iend = ibeg + ((Ny - 1)/Py) + (MyY < (Ny - 1)%Py);
    auto jbeg = MyX*((Nx - 1)/Px) + ( MyX < (Nx - 1)%Px ? MyX : (Nx - 1)%Px );
    auto jend = jbeg + ((Nx - 1)/Px) + (MyX < (Nx - 1)%Px);
    auto Nx_local = jend - jbeg + 1;
    for(auto i = ibeg+1; i < iend-1; ++i) {
        for(auto j = jbeg+1; j < jend-1; ++j) {
            const auto square = i*(Nx-1)+j;
            const auto whole_part = square/sum_k1_k2, remainder = square%sum_k1_k2;
            const auto i_local = i - ibeg;
            const auto j_local = j - jbeg;
            if((i*(Nx-1)+j) % sum_k1_k2 < k1){
                prev += 4;
                auto ind_cum_sum = whole_part * (k1 + 2*k2) + remainder;
                EN.cumulative_sum.push_back(prev);
                L2G.push_back(ind_cum_sum);
                EN.column_numbers.push_back(i_local*Nx_local+j_local);
                EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
                EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
                EN.column_numbers.push_back((i_local+1)*Nx_local+j_local+1);
            }
            else {
                prev += 3;
                auto ind_cum_sum = whole_part * (k1 + 2*k2) + (k1 + (remainder - k1) * 2);
                EN.cumulative_sum.push_back(prev);
                L2G.push_back(ind_cum_sum);
                EN.column_numbers.push_back(i_local*Nx_local+j_local);
                EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
                EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
                Part.push_back(MyID);

                prev += 3;
                L2G.push_back(ind_cum_sum+1);
                EN.cumulative_sum.push_back(prev);
                EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
                EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
                EN.column_numbers.push_back((i_local+1)*Nx_local+j_local+1);
            }
            Part.push_back(MyID);
        }
    }
    //~ EN.N_own = EN.cumulative_sum.size() - 1;
    EN.N = (jend - jbeg + 1)*(iend - ibeg + 1);
    //add interface or inner
    auto i = ibeg;
    for(auto j = jbeg; j < jend; ++j) {
        const auto square = i*(Nx-1)+j;
        const auto whole_part = square/sum_k1_k2, remainder = square%sum_k1_k2;
        const auto i_local = i - ibeg;
        const auto j_local = j - jbeg;
        if((i*(Nx-1)+j) % sum_k1_k2 < k1){
            prev += 4;
            auto ind_cum_sum = whole_part * (k1 + 2*k2) + remainder;
            EN.cumulative_sum.push_back(prev);
            L2G.push_back(ind_cum_sum);
            EN.column_numbers.push_back(i_local*Nx_local+j_local);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local+1);
        }
        else {
            prev += 3;
            auto ind_cum_sum = whole_part * (k1 + 2*k2) + (k1 + (remainder - k1) * 2);
            EN.cumulative_sum.push_back(prev);
            L2G.push_back(ind_cum_sum);
            EN.column_numbers.push_back(i_local*Nx_local+j_local);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            Part.push_back(MyID);

            prev += 3;
            L2G.push_back(ind_cum_sum+1);
            EN.cumulative_sum.push_back(prev);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local+1);
        }
        Part.push_back(MyID);
    }
    auto j = jend - 1;
    for(auto i = ibeg; i < iend; ++i) {
        const auto square = i*(Nx-1)+j;
        const auto whole_part = square/sum_k1_k2, remainder = square%sum_k1_k2;
        const auto i_local = i - ibeg;
        const auto j_local = j - jbeg;
        if((i*(Nx-1)+j) % sum_k1_k2 < k1){
            prev += 4;
            auto ind_cum_sum = whole_part * (k1 + 2*k2) + remainder;
            EN.cumulative_sum.push_back(prev);
            L2G.push_back(ind_cum_sum);
            EN.column_numbers.push_back(i_local*Nx_local+j_local);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local+1);
        }
        else {
            prev += 3;
            auto ind_cum_sum = whole_part * (k1 + 2*k2) + (k1 + (remainder - k1) * 2);
            EN.cumulative_sum.push_back(prev);
            L2G.push_back(ind_cum_sum);
            EN.column_numbers.push_back(i_local*Nx_local+j_local);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            Part.push_back(MyID);

            prev += 3;
            L2G.push_back(ind_cum_sum+1);
            EN.cumulative_sum.push_back(prev);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local+1);
        }
        Part.push_back(MyID);
    }
    i = iend;
    for(auto j = jbeg; j < jend; ++j) {
        const auto square = i*(Nx-1)+j;
        const auto whole_part = square/sum_k1_k2, remainder = square%sum_k1_k2;
        const auto i_local = i - ibeg;
        const auto j_local = j - jbeg;
        if((i*(Nx-1)+j) % sum_k1_k2 < k1){
            prev += 4;
            auto ind_cum_sum = whole_part * (k1 + 2*k2) + remainder;
            EN.cumulative_sum.push_back(prev);
            L2G.push_back(ind_cum_sum);
            EN.column_numbers.push_back(i_local*Nx_local+j_local);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local+1);
        }
        else {
            prev += 3;
            auto ind_cum_sum = whole_part * (k1 + 2*k2) + (k1 + (remainder - k1) * 2);
            EN.cumulative_sum.push_back(prev);
            L2G.push_back(ind_cum_sum);
            EN.column_numbers.push_back(i_local*Nx_local+j_local);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            Part.push_back(MyID);

            prev += 3;
            L2G.push_back(ind_cum_sum+1);
            EN.cumulative_sum.push_back(prev);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local+1);
        }
        Part.push_back(MyID);
    }
    j = jbeg;
    for(auto i = ibeg; i < iend; ++i) {
        const auto square = i*(Nx-1)+j;
        const auto whole_part = square/sum_k1_k2, remainder = square%sum_k1_k2;
        const auto i_local = i - ibeg;
        const auto j_local = j - jbeg;
        if((i*(Nx-1)+j) % sum_k1_k2 < k1){
            prev += 4;
            auto ind_cum_sum = whole_part * (k1 + 2*k2) + remainder;
            EN.cumulative_sum.push_back(prev);
            L2G.push_back(ind_cum_sum);
            EN.column_numbers.push_back(i_local*Nx_local+j_local);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local+1);
        }
        else {
            prev += 3;
            auto ind_cum_sum = whole_part * (k1 + 2*k2) + (k1 + (remainder - k1) * 2);
            EN.cumulative_sum.push_back(prev);
            L2G.push_back(ind_cum_sum);
            EN.column_numbers.push_back(i_local*Nx_local+j_local);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            Part.push_back(MyID);

            prev += 3;
            L2G.push_back(ind_cum_sum+1);
            EN.cumulative_sum.push_back(prev);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local+1);
        }
        Part.push_back(MyID);
    }
    //add halo up
    if(ibeg) {
        auto i = ibeg-1;
        for(auto j = jbeg; j < jend; ++j) {
            const auto square = i*(Nx-1)+j;
            const auto whole_part = square/sum_k1_k2, remainder = square%sum_k1_k2;
            const auto i_local = i - ibeg;
            const auto j_local = j - jbeg;
            if((i*(Nx-1)+j) % sum_k1_k2 < k1){
                auto ind_cum_sum = whole_part * (k1 + 2*k2) + remainder;
                L2G.push_back(ind_cum_sum);
            }
            else {
                auto ind_cum_sum = whole_part * (k1 + 2*k2) + (k1 + (remainder - k1) * 2);
                L2G.push_back(ind_cum_sum+1);
            }
            prev += 2;
            EN.cumulative_sum.push_back(prev);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local+1);
            Part.push_back(MyID - Px);
        }
    }
    //add halo right
    if(jend < Nx - 1) {
        auto j = jend;
        for(auto i = ibeg; i < iend; ++i) {
            const auto square = i*(Nx-1)+j;
            const auto whole_part = square/sum_k1_k2, remainder = square%sum_k1_k2;
            const auto i_local = i - ibeg;
            const auto j_local = j - jbeg;
            if((i*(Nx-1)+j) % sum_k1_k2 < k1){
                auto ind_cum_sum = whole_part * (k1 + 2*k2) + remainder;
                L2G.push_back(ind_cum_sum);
            }
            else {
                auto ind_cum_sum = whole_part * (k1 + 2*k2) + (k1 + (remainder - k1) * 2);
                L2G.push_back(ind_cum_sum);
            }
            prev += 2;
            EN.cumulative_sum.push_back(prev);
            EN.column_numbers.push_back(i_local*Nx_local+j_local);
            EN.column_numbers.push_back((i_local+1)*Nx_local+j_local);

            Part.push_back(MyID + 1);
        }
    }
    //add halo down
    if(iend < Ny - 1) {
        auto i = iend;
        for(auto j = jbeg; j < jend; ++j) {
            const auto square = i*(Nx-1)+j;
            const auto whole_part = square/sum_k1_k2, remainder = square%sum_k1_k2;
            const auto i_local = i - ibeg;
            const auto j_local = j - jbeg;
            if((i*(Nx-1)+j) % sum_k1_k2 < k1){
                auto ind_cum_sum = whole_part * (k1 + 2*k2) + remainder;
                L2G.push_back(ind_cum_sum);

            }
            else {
                auto ind_cum_sum = whole_part * (k1 + 2*k2) + (k1 + (remainder - k1) * 2);
                //~ EN.cumulative_sum.push_back(prev);
                L2G.push_back(ind_cum_sum);
            }
            prev += 2;
            EN.cumulative_sum.push_back(prev);
            EN.column_numbers.push_back(i_local*Nx_local+j_local);
            EN.column_numbers.push_back(i_local*Nx_local+j_local+1);
            Part.push_back(MyID + Px);
        }
    }
    //add halo left
    if(jbeg) {
        auto j = jbeg - 1;
        for(auto i = ibeg; i < iend; ++i) {
            const auto square = i*(Nx-1)+j;
            const auto whole_part = square/sum_k1_k2, remainder = square%sum_k1_k2;
            const auto i_local = i - ibeg;
            //~ const auto j_local = j - jbeg;
            if((i*(Nx-1)+j) % sum_k1_k2 < k1){
                auto ind_cum_sum = whole_part * (k1 + 2*k2) + remainder;
                L2G.push_back(ind_cum_sum);
            }
            else {
                auto ind_cum_sum = whole_part * (k1 + 2*k2) + (k1 + (remainder - k1) * 2);
                L2G.push_back(ind_cum_sum+1);
            }
            prev += 2;
            EN.cumulative_sum.push_back(prev);
            EN.column_numbers.push_back(i_local*Nx_local);
            EN.column_numbers.push_back((i_local+1)*Nx_local);
            Part.push_back(MyID - 1);
        }
    }
    EN.M = EN.cumulative_sum.size() - 1;
    return EN.reserved_memory();
}

//~ template <typename VarType /* тип значений */>
void Update(std::vector<double> &V, // Входной массив значений в вершинах/ячейках, который надо обновить
            //~ const tCommScheme &CS/*какая-то структура, описывающая схему обменов*/){
            tCommScheme &CS/*какая-то структура, описывающая схему обменов*/){
    const int B = CS.GetNumOfNeighbours(); // число соседей
    if(B==0) return; // нет соседей - нет проблем
    // tCommScheme - какая-то структура, замените ее на ваш вариант
    // приведем все к POD типам и неймингу, как было в тексте выше
    std::vector<int> Send = CS.GetSendList(); // список ячеек на отправку по всем соседям
    std::vector<int> Recv = CS.GetRecvList(); // список ячеек на прием по всем соседям
    std::vector<int> SendOffset = CS.GetSendOffset(); // смещения списков по каждому соседу на отправку
    std::vector<int> RecvOffset = CS.GetRecvOffset(); // смещения списков по каждому соседу на прием
    std::vector<int> Neighbours = CS.GetListOfNeighbours(); // номера процессов соседей
    //MPI_Comm MCW = CS.GetMyComm(); // коммуникатор для данной группы (MPI_COMM_WORLD)
    int sendCount=SendOffset[B]; // размер общего списка на отправку по всем соседям
    int recvCount=RecvOffset[B]; // размер общего списка на прием по всем соседям
    // MPI данные - сделаем статиками, поскольку это высокочастотная функция,
    // чтобы каждый раз не реаллокать (так делать вовсе не обязательно).
    static std::vector<double> SENDBUF, RECVBUF; // буферы на отправку и прием по всем соседям
    static std::vector<MPI_Request> REQ; // реквесты для неблокирующих обменов
    static std::vector<MPI_Status> STS; // статусы для неблокирующих обменов
    // ресайзим, если надо
    if(2*B > (int)REQ.size()){ REQ.resize(2*B); STS.resize(2*B); }
    if(sendCount>(int)SENDBUF.size()) SENDBUF.resize(sendCount);
    if(recvCount>(int)RECVBUF.size()) RECVBUF.resize(recvCount);
    int nreq=0; // сквозной счетчик реквестов сообщений
    // инициируем получение сообщений
    for(int p=0; p<B; p++){
        int SZ = (RecvOffset[p+1]-RecvOffset[p]);//*sizeof(VarType); // размер сообщения
        if(SZ<=0) continue; // если нечего слать - пропускаем соседа
        int NB_ID = Neighbours[p]; // узнаем номер процесса данного соседа
        int mpires = MPI_Irecv(&RECVBUF[RecvOffset[p]],//*sizeof(VarType)],
                                //SZ, MPI_CHAR, NB_ID, 0, MCW, &(REQ[nreq]));
                                SZ, MPI_DOUBLE, NB_ID, 0, MCW, &(REQ[nreq]));
        assert(mpires == MPI_SUCCESS);
        //~ ASSERT(mpires==MPI_SUCCESS, "MPI_Irecv failed");
        //ASSERT - какой-то макрос проверки-авоста, замените на ваш способ проверки
        nreq++;
    }
    // пакуем данные с интерфейса по единому списку сразу по всем соседям
    #pragma omp parallel for // в параллельном режиме с целью ускорения (К.О.)
    for(int i=0; i<sendCount; ++i) SENDBUF[i] = V[Send[i]/*номер ячейки на отправку*/];
    // инициируем отправку сообщений
    for(int p=0; p<B; p++){
        int SZ =(SendOffset[p+1]-SendOffset[p]);//*sizeof(VarType); // размер сообщения
        if(SZ<=0) continue; // если нечего принимать - пропускаем соседа
        int NB_ID = Neighbours[p]; // узнаем номер процесса данного соседа
        int mpires = MPI_Isend(&SENDBUF[SendOffset[p]],//*sizeof(VarType)],
                                //SZ, MPI_CHAR, NB_ID, 0, MCW, &(REQ[nreq]));
                                SZ, MPI_DOUBLE, NB_ID, 0, MCW, &(REQ[nreq]));
        assert(mpires == MPI_SUCCESS);
        //~ ASSERT(mpires==MPI_SUCCESS, "MPI_Isend failed");
        nreq++;
    }
    if(nreq>0){ // ждем завершения всех обменов
        int mpires = MPI_Waitall(nreq, &REQ[0], &STS[0]);
        assert(mpires == MPI_SUCCESS);
        //~ ASSERT(mpires==MPI_SUCCESS, "MPI_Waitall failed");
    }
    // разбираем данные с гало ячеек по единому списку сразу по всем соседям
    #pragma omp parallel for
    for(int i=0; i<recvCount; ++i) V[Recv[i]/*номер ячейки на прием*/] = RECVBUF[i];
}

int generate_grid_omp(int Nx, int Ny, int k1, int k2, CSRPortrait &EN) {

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
    EfE.N_own = EfE.M = EN.N_own;
    EfE.N = EN.M;
    EfE.cumulative_sum.clear();
    EfE.column_numbers.clear();
    EfE.cumulative_sum.resize(EfE.M+1, 0);
    std::vector<int> vector_to_intersect;
    for(auto i = 0; i < EN.N_own; ++i) {
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
              << "\t-Px <int> -Py <int> dividing the region into subdomains along the corresponding directions,\n"
              << "the product of Px and Py should be equal to the number of processes"
              << std::endl;
}

int initialization_A(const CSRPortrait &portrait, std::vector <double> &A_value, const std::vector <int> &L2G) {
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
                A_value[k] = std::cos(L2G[i] * L2G[j] + L2G[i] + L2G[j]);
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

int fill_in_b(std::vector <double> &b, int size_of_b, std::vector <int> &L2G) {
    b.clear();
    b.resize(size_of_b);
    #pragma omp parallel for
    for(auto i = 0; i < size_of_b; ++i) {
        b[i] = std::sin(L2G[i]);
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
         std::vector <double> &result_vector,
         int N) {

    #pragma omp parallel for
    for(auto i = 0; i < N; ++i) {
        result_vector[i] = vector_1[i] + scalar * vector_2[i];
    }

    return 0;
}

double dot(const std::vector <double> &vector_1,
           const std::vector <double> &vector_2,
           int N) {
    double result = 0.;
    #pragma omp parallel for reduction(+ : result)
    for(auto i = 0; i < N; ++i) {
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
                    portrait_M.cumulative_sum[i+1] = i+1;
                }
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
          int N,
          tCommScheme &CS) {
    CSRPortrait portrait_M;
    std::vector <double> M_value;
    construct_matrix_M(portrait_A, A_value, portrait_M, M_value);
    double rho_k = 0, rho_k_1 = 0, eps_q = eps*eps;
    std::vector <double> r = b; //    𝒓0 = 𝒃

    int k = 0;                  //    𝑘 = 0
    int N_own = portrait_A.N_own;
    x.clear();
    x.resize(N, 0);  //    𝒙0 = 0
    std::vector <double> z(N);
    std::vector <double> p(N);
    std::vector <double> q(N);
    do {
        ++k;
        Update(r,CS);
        SpMV(portrait_M, M_value, r, z); //𝒛𝑘 = 𝑴−1𝒓𝑘−1 // SpMV

        rho_k = dot(r, z, N_own);               //𝜌𝑘 = (𝒓𝑘−1, 𝒛𝑘) // dot
        MPI_Allreduce(MPI_IN_PLACE, &rho_k, 1, MPI_DOUBLE, MPI_SUM, MCW);

        if (k == 1) {                    //if 𝑘 = 1 then
            p = z;                       //𝒑𝑘 = 𝒛𝑘
        } else {
            double beta = rho_k / rho_k_1; //𝛽𝑘 = 𝜌𝑘/𝜌𝑘−1
            axpy(z, p, beta, p, N_own);         //𝒑𝑘 = 𝒛𝑘 + 𝛽𝑘𝒑𝑘−1 // axpy
        }
        Update(p,CS);
        SpMV(portrait_A, A_value, p, q); //𝒒𝑘 = 𝑨𝒑𝑘 // SpMV
        double rho_q = dot(p, q, N_own);
        MPI_Allreduce(MPI_IN_PLACE, &rho_q, 1, MPI_DOUBLE, MPI_SUM, MCW);
        double alpha = rho_k / rho_q;  //𝛼𝑘 = 𝜌𝑘/(𝒑𝑘, 𝒒𝑘) // dot
        axpy(x, p, alpha, x, N_own);            //𝒙𝑘 = 𝒙𝑘−1 + 𝛼𝑘𝒑𝑘 // axpy
        axpy(r, q, alpha*(-1), r, N_own);       //𝒓𝑘 = 𝒓𝑘−1 − 𝛼𝑘𝒒𝑘 // axpy
        rho_k_1 = rho_k;
        if(MyID == MASTER_ID && k%10 == 0) {
            std::cout << "Iteration: " << k << ", rho: " << rho_k << std::endl;
        }

    } while (rho_k > eps_q && k < maxit);//𝜌𝑘 > 𝜀^2 and k < maxit

    return k;
}

int args_parsing(int argc, char *argv[], std::vector <int> &args_values, double &epsilon) {
    args_values.clear();

    int Nx = 0, Ny = 0, k1 = 0, k2 = 0, Py = 0, Px = 0;
    double eps = 0.;
    bool debug_mode = false, args_exist[] = {false,false,false,false};
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "-Nx") {
            if (i + 1 < argc) {
                Nx = strtol(argv[++i], nullptr, 0);
                if(Nx <= 1 || Nx >= MAX_SIZE) {
                    if(MyID == MASTER_ID)
                        std::cerr << "-Nx value most be greater then 1 and less than " << MAX_SIZE << std::endl;
                    exit(1);
                }
                args_exist[0] = true;
            } else {
                if(MyID == MASTER_ID)
                    std::cerr << "-Nx option requires one argument." << std::endl;
                exit(1);
            }
        } else if(arg == "-Ny") {
            if (i + 1 < argc) {
                Ny = strtol(argv[++i], nullptr, 0);
                if(Ny <= 1 || Ny >= MAX_SIZE) {
                    if(MyID == MASTER_ID)
                        std::cerr << "-Ny value most be greater then 1 and less than " << MAX_SIZE << std::endl;
                    exit(2);
                }
                args_exist[1] = true;
            } else {
                if(MyID == MASTER_ID)
                    std::cerr << "-Ny option requires one argument." << std::endl;
                exit(2);
            }
        } else if(arg == "-k1") {
            if (i + 1 < argc) {
                k1 = strtol(argv[++i], nullptr, 0);
                if(k1 < 0) {
                    if(MyID == MASTER_ID)
                        std::cerr << "-k1 value must be at least 0 " <<std::endl;
                    exit(3);
                }
                args_exist[2] = true;
            } else {
                if(MyID == MASTER_ID)
                    std::cerr << "-k1 value requires one argument." << std::endl;
                exit(3);
            }
        } else if(arg == "-k2") {
            if (i + 1 < argc) {
                k2 = strtol(argv[++i], nullptr, 0);
                if(k2 < 0) {
                    if(MyID == MASTER_ID)
                        std::cerr << "-k2 value must be at least 0 " <<std::endl;
                    exit(4);
                }
                args_exist[3] = true;
            } else {
                if(MyID == MASTER_ID)
                    std::cerr << "-k2 option requires one argument." << std::endl;
                exit(4);
            }
        } else if(arg == "-d" || arg == "--debug") {
            debug_mode = true;
        } else if(arg == "-h"||arg == "--help") {
            if(MyID == MASTER_ID)
                show_usage(argv[0]);
            exit(0);
        } else if(arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                auto num_threads = strtol(argv[++i], nullptr, 0);
                if((num_threads < 1 || num_threads >= MAX_SIZE) && (MyID == MASTER_ID)) {
                    std::cerr << "-t value most be greater then 1 and less than " << MAX_SIZE << std::endl;
                    std::cerr << "-t value will be default " << std::endl;
                } else {
                    omp_set_num_threads(num_threads);
                }
            }
        }  else if(arg == "-Px") {
            if (i + 1 < argc) {
                Px = strtol(argv[++i], nullptr, 0);
            }
        } else if(arg == "-Py") {
            if (i + 1 < argc) {
                Py = strtol(argv[++i], nullptr, 0);
            }
        } else if(arg == "-e" || "--epsilon") {
            if (i + 1 < argc) {
                eps = strtod(argv[++i], nullptr);
            }
        }
    }
    if(eps <= 0.) {
        if(MyID == MASTER_ID) {
            std::cerr << "-e value most be greater then 0." <<std::endl;
            std::cerr << "-e value will be default" <<std::endl;
        }
        eps = 1e-6;
    }
    if(debug_mode && Nx*Ny > 100) {
        if(MyID == MASTER_ID)
            std::cout << "The size of the matrix is too large to display on the screen, debugging mode is disabled" << std::endl;
        debug_mode = false;
    }
    if(args_exist[0]*args_exist[1]*args_exist[2]*args_exist[3]*Nx*Ny == 0 || k1+k2 == 0) {
        if(MyID == MASTER_ID) {
            std::cout << "Sorry, several required arguments are not specified or k1+k2=0" << std::endl;
            std::cout << "Try to read help." <<std::endl;
            show_usage(argv[0]);
        }
        exit(5);
    }
    if(NumProc != Px*Py) {
        if(MyID == MASTER_ID) {
            std::cout << "Px * Py != number of processes" << std::endl;
        }
        MPI_Finalize();
        exit(666);
    }
    args_values.resize(7);
    args_values[0] = Nx;
    args_values[1] = Ny;
    args_values[2] = k1;
    args_values[3] = k2;
    args_values[4] = debug_mode;
    epsilon = eps;
    args_values[5] = Px;
    args_values[6] = Py;

    return 0;
}

int main(int argc, char *argv[]) {

    if (auto rc = MPI_Init(&argc, &argv)) {
        printf("Startup error, execution stopped\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &NumProc);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyID);

    std::vector <int> args_values;
    double eps = 1e-6;
    args_parsing(argc, argv, args_values, eps);
    int Nx = args_values[0], Ny = args_values[1], k1 = args_values[2], k2 = args_values[3];
    bool debug_mode = args_values[4];
    int Px = args_values[5], Py = args_values[6];

    CSRPortrait newcsr, tcsr;
    std::vector <int> L2G;
    std::vector <int> Part;
    MPI_Barrier(MCW);
    double start = MPI_Wtime();
    auto occupied_memory = generate_grid(Nx, Ny, k1, k2, newcsr, Px, Py, L2G, Part);
    MPI_Barrier(MCW);
    double stop = MPI_Wtime();
    int N = newcsr.M;
    if(MyID == MASTER_ID) {
        std::cout << "Grid generation time:                      " << stop - start << std::endl;
        std::cout << "Used memory for grid generation in master: " << occupied_memory << " bytes" << std::endl;
    }
    MPI_Barrier(MCW);
    start = MPI_Wtime();
    occupied_memory = construct_matrix_E_f_E(newcsr, tcsr);
    MPI_Barrier(MCW);
    stop = MPI_Wtime();
    if(MyID == MASTER_ID) {
        std::cout << "EfE matrix generation time:                " << stop - start << std::endl;
        std::cout << "Used memory for EfE generation in master:  " << occupied_memory << " bytes" << std::endl;
    }

    auto N_own = tcsr.N_own;    //число собственных элементов для процесса

    std::vector <double> A_value;
    MPI_Barrier(MCW);
    start = MPI_Wtime();
    initialization_A(tcsr, A_value, L2G);
    MPI_Barrier(MCW);
    stop = MPI_Wtime();

    if(MyID == MASTER_ID) {
        std::cout << "Matrix values generation time:   " << stop - start << std::endl;
    }

    std::vector <double> b;

    MPI_Barrier(MCW);
    start = MPI_Wtime();
    fill_in_b(b, N, L2G);
    MPI_Barrier(MCW);
    stop = MPI_Wtime();

    if(MyID == MASTER_ID) {
        std::cout << "Vector b values generation time: " << stop - start << std::endl;
    }

    tCommScheme CS(N, tcsr, Part, L2G);

    std::vector <double> x;
    int maxit = 100;
    MPI_Barrier(MCW);
    start = MPI_Wtime();
    auto iter_count = solve(tcsr, A_value, b, eps, maxit, x, N, CS);
    MPI_Barrier(MCW);
    stop = MPI_Wtime();

    std::vector <double> z(x.size());
    Update(x,CS);
    SpMV(tcsr, A_value, x, z);
    axpy(z, b, -1, z, N_own);
    double res = dot(z,z,N_own);
    MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_SUM, MCW);
    if(MyID == MASTER_ID) {
        std::cout << "Solve time:                 " << stop - start << std::endl;
        std::cout << "Epsilon:                    " << eps << std::endl;
        std::cout << "Total number of iterations: " << iter_count << ", ||Ax - b||^2: "
        << res << std::endl;
    }

    if(debug_mode) {
        for(auto i = 0; i < NumProc; ++i) {
            MPI_Barrier(MCW);
            if(MyID == i) {

                std::cout << "MyID: " << MyID << " Matrix EfE:" << std::endl;
                print_CSRPortrait(tcsr);
                std::cout << "Neighbors count: " << CS.GetNumOfNeighbours() << std::endl;
                std::cout << "LON:\n {";
                for(auto x : CS.GetListOfNeighbours()) {
                    std::cout << x << " ";
                }
                std::cout << "}\n";

                std::cout << "Recive list:\n {";
                for(auto x : CS.GetRecvList()) {
                    std::cout << x << " ";
                }
                std::cout << "}\n";

                std::cout << "Recive offsets:\n {";
                for(auto x : CS.GetRecvOffset()) {
                    std::cout << x << " ";
                }
                std::cout << "}\n";

                std::cout << "Send list:\n {";
                for(auto x : CS.GetSendList()) {
                    std::cout << x << " ";
                }
                std::cout << "}\n";

                std::cout << "Send offsets:\n {";
                for(auto x : CS.GetSendOffset()) {
                    std::cout << x << " ";
                }
                std::cout << "}\n";

                std::cout << "Part:\n {";
                for(auto x : Part) {
                    std::cout << x << " ";
                }
                std::cout << "}\n";

                std::cout << "Local 2 Global:\n {";
                for(auto x : L2G) {
                    std::cout << x << " ";
                }
                std::cout << "}\n";

                std::cout << "x:\n {";
                for(auto x : x) {
                    std::cout << x << " ";
                }
                std::cout << "}\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
